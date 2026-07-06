// itoa benchmark driver.
//
// Beyond the original "one fixed digit length at a time" (sequential) and
// "shuffle of all lengths" (random) cases, this driver adds:
//
//   * bylength   - throughput for values that all have exactly D digits,
//                  swept over every D. (The predictable / best case.)
//   * loguniform - a shuffled set with an equal number of values in each
//                  digit-length bin, over a window of lengths (or all
//                  lengths). "Log uniform" == uniform over the number of
//                  digits == uniform in log10(value).
//   * admixture  - a shuffled mix of just two digit lengths (default 5 and 6)
//                  at a controlled ratio (0%,10%,...,100%). Sweeping the ratio
//                  isolates the cost of branch misprediction on the
//                  digit-count branch: pure mixes predict perfectly, 50/50
//                  maximises mispredicts.
//
// Every mode runs for the full set of widths, signed and unsigned in parallel:
// u32/i32, u64/i64, and u128/i128. 128-bit conversion has no standard
// formatter, so only the implementations that opt in (fmt, naive, null)
// provide it; the rest are skipped for 128-bit automatically.
//
// Measurement uses an ABBA / interleaved engine: within each round every
// implementation is timed once over the same dataset, and the roster order is
// reversed every other round. Timing an implementation in both early and late
// slots balances any monotonic drift (turbo ramp, thermal) across the roster,
// and the reported figure is the minimum over rounds.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "resultfilename.h"
#include "test.h"

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// Prevent the optimiser from discarding the formatted buffer.
// ---------------------------------------------------------------------------
static inline void Escape(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

// Clear the upper 128 bits of the YMM registers. An AVX (256-bit) routine that
// leaves the upper state "dirty" imposes a per-instruction transition penalty on
// subsequent legacy-SSE code. Because the engine times different implementations
// back to back (an AVX one right before a legacy-SSE one), we zero the upper
// state between timed blocks so that penalty is not misattributed.
static inline void ClearUpper() {
#if defined(__AVX__)
    _mm256_zeroupper();
#endif
}

// ---------------------------------------------------------------------------
// Per-type traits: buffer size, max digit count, sign, name, and range.
// numeric_limits is not portable for __int128, so the range lives here.
// ---------------------------------------------------------------------------
template <typename T> struct Traits;

// Buffer sizes are generous: some SIMD implementations over-store (writing a
// fixed width regardless of digit count, plus a sign byte), so leave headroom.
template <> struct Traits<uint32_t> {
    enum { kBufferSize = 24, kMaxDigit = 10, kSigned = 0 };
    static const char* Name() { return "u32"; }
    static uint128_t MaxMagnitude() { return UINT32_MAX; }
    static uint32_t Min() { return 0; }
    static uint32_t Max() { return UINT32_MAX; }
};
template <> struct Traits<int32_t> {
    enum { kBufferSize = 24, kMaxDigit = 10, kSigned = 1 };
    static const char* Name() { return "i32"; }
    static uint128_t MaxMagnitude() { return INT32_MAX; }
    static int32_t Min() { return INT32_MIN; }
    static int32_t Max() { return INT32_MAX; }
};
template <> struct Traits<uint64_t> {
    enum { kBufferSize = 32, kMaxDigit = 20, kSigned = 0 };
    static const char* Name() { return "u64"; }
    static uint128_t MaxMagnitude() { return UINT64_MAX; }
    static uint64_t Min() { return 0; }
    static uint64_t Max() { return UINT64_MAX; }
};
template <> struct Traits<int64_t> {
    enum { kBufferSize = 32, kMaxDigit = 19, kSigned = 1 };
    static const char* Name() { return "i64"; }
    static uint128_t MaxMagnitude() { return INT64_MAX; }
    static int64_t Min() { return INT64_MIN; }
    static int64_t Max() { return INT64_MAX; }
};
template <> struct Traits<uint128_t> {
    enum { kBufferSize = 64, kMaxDigit = 39, kSigned = 0 };
    static const char* Name() { return "u128"; }
    static uint128_t MaxMagnitude() { return ~(uint128_t)0; }
    static uint128_t Min() { return 0; }
    static uint128_t Max() { return ~(uint128_t)0; }
};
template <> struct Traits<int128_t> {
    enum { kBufferSize = 64, kMaxDigit = 39, kSigned = 1 };
    static const char* Name() { return "i128"; }
    static uint128_t MaxMagnitude() { return (~(uint128_t)0) >> 1; }
    static int128_t Min() { return (int128_t)((uint128_t)1 << 127); }
    static int128_t Max() { return (int128_t)((~(uint128_t)0) >> 1); }
};

// ---------------------------------------------------------------------------
// Map a width to the corresponding function pointer stored in a Test.
// ---------------------------------------------------------------------------
template <typename T> struct FnAccess;
template <> struct FnAccess<uint32_t>  { static auto Get(const Test* t) { return t->u32toa;  } };
template <> struct FnAccess<int32_t>   { static auto Get(const Test* t) { return t->i32toa;  } };
template <> struct FnAccess<uint64_t>  { static auto Get(const Test* t) { return t->u64toa;  } };
template <> struct FnAccess<int64_t>   { static auto Get(const Test* t) { return t->i64toa;  } };
template <> struct FnAccess<uint128_t> { static auto Get(const Test* t) { return t->u128toa; } };
template <> struct FnAccess<int128_t>  { static auto Get(const Test* t) { return t->i128toa; } };

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------
static uint128_t Pow10(int e) {  // e must be <= 38 (10^39 overflows 128 bits)
    uint128_t r = 1;
    while (e-- > 0)
        r *= 10;
    return r;
}

// Inclusive magnitude range [lo, hi] of values with exactly d decimal digits,
// clamped to what type T can represent.
template <typename T>
static void DigitRange(int d, uint128_t& lo, uint128_t& hi) {
    lo = (d <= 1) ? 1 : Pow10(d - 1);
    const uint128_t maxMag = Traits<T>::MaxMagnitude();
    hi = (d >= Traits<T>::kMaxDigit) ? maxMag : (Pow10(d) - 1);
    if (hi > maxMag) hi = maxMag;
}

static uint128_t RandRange(std::mt19937_64& rng, uint128_t lo, uint128_t hi) {
    const uint128_t span = hi - lo;              // width - 1
    if (span == 0) return lo;
    uint128_t r = ((uint128_t)rng() << 64) | (uint128_t)rng();
    return lo + r % (span + 1);                  // span+1 cannot overflow here
}

template <typename T>
static T MakeValue(uint128_t magnitude, bool negative) {
    T v = (T)magnitude;
    if (Traits<T>::kSigned && negative)
        v = (T)(-v);
    return v;
}

// A dataset value at digit length d with a (deterministic) random sign.
template <typename T>
static T RandomValueOfLength(std::mt19937_64& rng, int d) {
    uint128_t lo, hi;
    DigitRange<T>(d, lo, hi);
    bool neg = Traits<T>::kSigned && (rng() & 1);
    return MakeValue<T>(RandRange(rng, lo, hi), neg);
}

// ---------------------------------------------------------------------------
// Dataset generators
// ---------------------------------------------------------------------------
template <typename T>
static std::vector<T> GenByLength(std::mt19937_64& rng, int d, size_t n) {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = RandomValueOfLength<T>(rng, d);
    return v;
}

template <typename T>
static std::vector<T> GenLogUniform(std::mt19937_64& rng, int loD, int hiD, size_t n) {
    const int bins = hiD - loD + 1;
    const size_t per = std::max<size_t>(1, n / bins);
    std::vector<T> v;
    v.reserve(per * bins);
    for (int d = loD; d <= hiD; d++)
        for (size_t i = 0; i < per; i++)
            v.push_back(RandomValueOfLength<T>(rng, d));
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

// Equidistributed values in [lo, hi] (default bytes, 0..255): uniform over
// VALUES, not digit lengths, so short lengths dominate. No sign flip -- this
// models byte-like data. Already i.i.d., no shuffle needed.
template <typename T>
static std::vector<T> GenUniform(std::mt19937_64& rng, uint64_t lo, uint64_t hi, size_t n) {
    uint128_t hiC = hi;
    if (hiC > Traits<T>::MaxMagnitude())
        hiC = Traits<T>::MaxMagnitude();
    std::vector<T> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = (T)RandRange(rng, lo, hiC);
    return v;
}

// A "mixed" set of n values cycling through every digit length 1..kMaxDigit,
// then shuffled. This is the predictor-thrashing noise used by the
// unpredictable mode (mirrors dtolnay's itoa-benchmark).
template <typename T>
static std::vector<T> GenMixedAllLengths(std::mt19937_64& rng, size_t n) {
    std::vector<T> v(n);
    const int N = Traits<T>::kMaxDigit;
    for (size_t i = 0; i < n; i++)
        v[i] = RandomValueOfLength<T>(rng, (int)(i % N) + 1);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

template <typename T>
static std::vector<T> GenAdmixture(std::mt19937_64& rng, int dA, int dB, int percentA, size_t n) {
    const size_t countA = (size_t)((n * (uint64_t)percentA) / 100);
    std::vector<T> v;
    v.reserve(n);
    for (size_t i = 0; i < countA; i++)     v.push_back(RandomValueOfLength<T>(rng, dA));
    for (size_t i = countA; i < n; i++)     v.push_back(RandomValueOfLength<T>(rng, dB));
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

// Admixture straddling a threshold (only meaningful for 128-bit T). zmij's
// 128-bit path has two internal branches, both of which cost a mispredict on a
// mixed stream but neither of which a digit-count admixture (5,6...) reaches:
//   * value <= UINT64_MAX (2^64): delegate to the u64 path vs the chunked u128
//     path -- a boundary in *bits*, mid-way through the 20-digit range.
//   * value < 1e32: one 16-digit peel (20-32 digits) vs two peels (33-39) --
//     the interior-chunk count branch.
// This mixes values ~2 decimal orders below the threshold with ~2 orders above,
// so sweeping the ratio exposes the chosen branch regardless of digit counting.
template <typename T>
static std::vector<T> GenStraddle(std::mt19937_64& rng, uint128_t threshold,
                                  int percentBelow, size_t n) {
    const uint128_t loBelow = threshold / 100, hiBelow = threshold - 1;      // below
    const uint128_t loAbove = threshold,       hiAbove = threshold * 100 - 1;  // above
    const size_t countBelow = (size_t)((n * (uint64_t)percentBelow) / 100);
    std::vector<T> v;
    v.reserve(n);
    for (size_t i = 0; i < n; i++) {
        bool below = i < countBelow;
        uint128_t mag = below ? RandRange(rng, loBelow, hiBelow)
                              : RandRange(rng, loAbove, hiAbove);
        bool neg = Traits<T>::kSigned && (rng() & 1);
        v.push_back(MakeValue<T>(mag, neg));
    }
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

// ---------------------------------------------------------------------------
// Configuration (populated from argv)
// ---------------------------------------------------------------------------
struct AdmixPair { int a, b; };

struct Config {
    std::set<std::string> modes { "bylength", "loguniform", "unpredictable", "admixture", "uniform" };
    std::set<std::string> types { "u32", "i32", "u64", "i64", "u128", "i128" };
    std::vector<std::string> filters;                  // empty => all
    std::vector<AdmixPair> admixPairs { { 5, 6 } };
    int admixStep = 10;
    // loguniform windows, empty => full range per type
    std::vector<std::pair<int,int>> logWindows;
    // uniform-mode value range (equidistributed values, byte-like data)
    uint64_t uniformLo = 0, uniformHi = 255;

    // Dataset size (values). Large on purpose: the branch-sensitive modes
    // replay a fixed shuffled sequence, and a modern TAGE predictor memorizes
    // short ones (a single-branch impl like jeaiii shows *zero* admixture hump
    // at 4096 vs its true ~+160% at 64K+). 64Ki defeats the predictor while
    // keeping even u128 (1MiB) in L3. Runtime is ~size-independent: passes
    // auto-scales as passTarget/size, so total conversions stay constant.
    size_t size = 65536;
    unsigned rounds = 6;         // ABBA rounds
    uint64_t passTarget = 1u << 20;  // ~ calls per single timing
    bool verify = true;
    std::string out;             // csv path, empty => auto

    bool WantType(const char* n) const { return types.count(n) != 0; }
    bool MatchFilter(const char* name) const {
        if (filters.empty()) return true;
        for (const std::string& f : filters)
            if (strstr(name, f.c_str())) return true;
        return false;
    }
};

// ---------------------------------------------------------------------------
// Verification (compares each implementation against naive)
// ---------------------------------------------------------------------------
template <typename T>
static void VerifyValue(T value, void(*f)(T, char*), void(*g)(T, char*),
                        const char* fname, const char* gname) {
    char buffer1[Traits<T>::kBufferSize] = {};
    char buffer2[Traits<T>::kBufferSize] = {};
    f(value, buffer1);
    g(value, buffer2);
    // yy copies 2 bytes at a time and does not always null-terminate; trim to
    // the reference length before comparing.
    if (strcmp(gname, "yy") == 0)
        buffer2[strlen(buffer1)] = '\0';
    if (strcmp(buffer1, buffer2) != 0) {
        printf("\nError: %s -> %s, %s -> %s\n", fname, buffer1, gname, buffer2);
        throw std::exception();
    }
}

template <typename T>
static void Verify(void(*f)(T, char*), void(*g)(T, char*),
                   const char* fname, const char* gname) {
    printf("Verifying %-14s %s = %s ... ", Traits<T>::Name(), fname, gname);

    VerifyValue<T>(0, f, g, fname, gname);
    VerifyValue<T>(Traits<T>::Min(), f, g, fname, gname);
    VerifyValue<T>(Traits<T>::Max(), f, g, fname, gname);

    for (uint32_t power = 2; power <= 10; power += 8) {
        T i = 1;
        for (;;) {
            VerifyValue<T>(i - 1, f, g, fname, gname);
            VerifyValue<T>(i, f, g, fname, gname);
            if (Traits<T>::kSigned) {
                VerifyValue<T>((T)-i, f, g, fname, gname);
                VerifyValue<T>((T)-(i + 1), f, g, fname, gname);
            }
            // Overflow-proof advance: signed i *= power past max is UB, which
            // clang -O3 turns into an infinite loop (gcc happens to wrap).
            if (i > Traits<T>::Max() / (T)power)
                break;
            i *= power;
        }
    }
    printf("OK\n");
}

template <typename T>
static void VerifyWidth(const Test* naive, const Test* impl) {
    auto nf = FnAccess<T>::Get(naive);
    auto gf = FnAccess<T>::Get(impl);
    if (!nf || !gf) return;            // width unsupported by one of them
    Verify<T>(nf, gf, "naive", impl->fname);
}

static void VerifyAll(const Config& cfg) {
    const TestList& tests = TestManager::Instance().GetTests();
    const Test* naive = nullptr;
    for (const Test* t : tests)
        if (strcmp(t->fname, "naive") == 0) { naive = t; break; }
    assert(naive != nullptr);

    for (const Test* t : tests) {
        if (strcmp(t->fname, "null") == 0) continue;
        // toy256 is by design only correct on [0, 255] (it truncates), so it
        // cannot pass the general verification.
        if (strcmp(t->fname, "toy256") == 0) continue;
        if (!cfg.MatchFilter(t->fname)) continue;
        try {
            if (cfg.WantType("u32"))  VerifyWidth<uint32_t>(naive, t);
            if (cfg.WantType("i32"))  VerifyWidth<int32_t>(naive, t);
            if (cfg.WantType("u64"))  VerifyWidth<uint64_t>(naive, t);
            if (cfg.WantType("i64"))  VerifyWidth<int64_t>(naive, t);
            if (cfg.WantType("u128")) VerifyWidth<uint128_t>(naive, t);
            if (cfg.WantType("i128")) VerifyWidth<int128_t>(naive, t);
        } catch (...) {
        }
    }
}

// ---------------------------------------------------------------------------
// ABBA / interleaved measurement engine
// ---------------------------------------------------------------------------
template <typename T>
struct FuncEntry { const char* name; void (*fn)(T, char*); };

template <typename T>
static std::vector<double> InterleavedMeasure(const std::vector<FuncEntry<T>>& fns,
                                              const std::vector<T>& data,
                                              const Config& cfg) {
    const size_t n = data.size();
    const unsigned passes = (unsigned)std::max<uint64_t>(1, cfg.passTarget / std::max<size_t>(1, n));
    char buffer[Traits<T>::kBufferSize];
    std::vector<double> best(fns.size(), std::numeric_limits<double>::max());

    // Warm-up: touch every function once over the dataset (untimed) so the first
    // timed round is not cold (I$/D$/branch predictor primed, turbo ramped).
    for (size_t i = 0; i < fns.size(); i++) {
        for (size_t k = 0; k < n; k++)
            fns[i].fn(data[k], buffer);
        Escape(buffer);
        ClearUpper();
    }

    for (unsigned r = 0; r < cfg.rounds; r++) {
        const bool forward = (r % 2 == 0);
        for (size_t j = 0; j < fns.size(); j++) {
            const size_t i = forward ? j : (fns.size() - 1 - j);
            void (*f)(T, char*) = fns[i].fn;

            ClearUpper();   // start each timed block with clean YMM upper state
            auto t0 = std::chrono::steady_clock::now();
            for (unsigned p = 0; p < passes; p++) {
                for (size_t k = 0; k < n; k++)
                    f(data[k], buffer);
                Escape(buffer);
            }
            auto t1 = std::chrono::steady_clock::now();

            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count()
                        / (double)((uint64_t)passes * n);
            best[i] = std::min(best[i], ns);
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
template <typename T>
static void Report(FILE* fp, const std::vector<FuncEntry<T>>& fns,
                   const std::vector<double>& res,
                   const char* mode, const char* series, long x) {
    for (size_t i = 0; i < fns.size(); i++)
        fprintf(fp, "%s,%s,%s,%s,%ld,%f\n",
                Traits<T>::Name(), fns[i].name, mode, series, x, res[i]);
}

// ---------------------------------------------------------------------------
// Run all requested modes for a single width
// ---------------------------------------------------------------------------
static uint64_t gSeed = 20260704ull;   // fixed => reproducible datasets

template <typename T>
static void RunType(const Config& cfg, FILE* fp) {
    if (!cfg.WantType(Traits<T>::Name())) return;

    std::vector<FuncEntry<T>> fns;
    for (const Test* t : TestManager::Instance().GetTests()) {
        auto fn = FnAccess<T>::Get(t);
        if (!fn) continue;                       // no support for this width
        if (!cfg.MatchFilter(t->fname)) continue;
        fns.push_back({ t->fname, fn });
    }
    if (fns.empty()) return;

    // toy256 truncates to [0, 255], so it is only meaningful in the byte-range
    // 'uniform' regime; every other mode uses the list without it.
    std::vector<FuncEntry<T>> fnsNoToy;
    for (const auto& e : fns)
        if (strcmp(e.name, "toy256") != 0) fnsNoToy.push_back(e);

    std::mt19937_64 rng(gSeed);
    const char* tn = Traits<T>::Name();
    printf("\n==== %-4s  (%zu impls) ====\n", tn, fns.size());

    // ---- bylength -------------------------------------------------------
    if (cfg.modes.count("bylength")) {
        printf("  bylength (ns/op by digit count):\n");
        for (int d = 1; d <= Traits<T>::kMaxDigit; d++) {
            auto data = GenByLength<T>(rng, d, cfg.size);
            auto res = InterleavedMeasure<T>(fnsNoToy, data, cfg);
            Report<T>(fp, fnsNoToy, res, "bylength", "", d);
            double lo = *std::min_element(res.begin(), res.end());
            printf("    d=%2d  best=%7.3f ns\n", d, lo);
        }
    }

    // ---- loguniform -----------------------------------------------------
    if (cfg.modes.count("loguniform")) {
        std::vector<std::pair<int,int>> windows = cfg.logWindows;
        if (windows.empty()) {
            windows.push_back({ 1, Traits<T>::kMaxDigit });   // all lengths
            windows.push_back({ 1, 4 });                      // "realistic": small values
        }
        for (auto& w : windows) {
            int loD = std::max(1, w.first);
            int hiD = std::min((int)Traits<T>::kMaxDigit, w.second);
            if (loD > hiD) continue;
            auto data = GenLogUniform<T>(rng, loD, hiD, cfg.size);
            auto res = InterleavedMeasure<T>(fnsNoToy, data, cfg);
            char series[32];
            snprintf(series, sizeof series, "%d-%d", loD, hiD);
            Report<T>(fp, fnsNoToy, res, "loguniform", series, hiD - loD + 1);
            double lo = *std::min_element(res.begin(), res.end());
            printf("  loguniform[%s] (%d lengths): best=%7.3f ns\n",
                   series, hiD - loD + 1, lo);
        }
    }

    // ---- uniform ---------------------------------------------------------
    // Equidistributed values in a small range (default 0..255): the byte-
    // printing regime, where a full string-lookup table (toy256) is possible.
    if (cfg.modes.count("uniform")) {
        auto data = GenUniform<T>(rng, cfg.uniformLo, cfg.uniformHi, cfg.size);
        auto res = InterleavedMeasure<T>(fns, data, cfg);
        char series[48];
        snprintf(series, sizeof series, "%llu-%llu",
                 (unsigned long long)cfg.uniformLo,
                 (unsigned long long)cfg.uniformHi);
        Report<T>(fp, fns, res, "uniform", series, 0);
        double lo = *std::min_element(res.begin(), res.end());
        printf("  uniform[%s]: best=%7.3f ns\n", series, lo);
    }

    // ---- unpredictable (dtolnay-style) ----------------------------------
    // Marginal cost of a length-d value amid random-length "noise": measure
    // (noise ++ length-d) and subtract the noise-only baseline.
    //
    // This matches dtolnay's algorithm exactly, but note the name is a
    // misnomer: the combined set is HALF one constant length (the size copies
    // of length-d appended to size noise values), so >50% of the stream is
    // predictable and the predictor biases toward length-d and predicts those
    // values correctly. It therefore UNDER-reports branch-misprediction cost --
    // e.g. jeaiii reads near its predicted-compute floor here while the
    // admixture sweep shows its true ~+6ns hump. Use admixture for branch cost;
    // this mode is kept only for comparability with dtolnay's benchmark.
    if (cfg.modes.count("unpredictable")) {
        auto mixed = GenMixedAllLengths<T>(rng, cfg.size);
        auto baseline = InterleavedMeasure<T>(fnsNoToy, mixed, cfg);   // ns/op over noise
        printf("  unpredictable (marginal ns/op by digit count, noise-subtracted):\n");
        for (int d = 1; d <= Traits<T>::kMaxDigit; d++) {
            std::vector<T> data = mixed;
            for (size_t i = 0; i < cfg.size; i++)
                data.push_back(RandomValueOfLength<T>(rng, d));
            std::shuffle(data.begin(), data.end(), rng);
            auto comb = InterleavedMeasure<T>(fnsNoToy, data, cfg);    // ns/op over 2*size
            // marginal per length-d op = 2*comb - baseline (see note in header)
            std::vector<double> marg(fnsNoToy.size());
            for (size_t i = 0; i < fnsNoToy.size(); i++)
                marg[i] = 2.0 * comb[i] - baseline[i];
            Report<T>(fp, fnsNoToy, marg, "unpredictable", "", d);
            double lo = *std::min_element(marg.begin(), marg.end());
            printf("    d=%2d  best=%7.3f ns\n", d, lo);
        }
    }

    // ---- admixture ------------------------------------------------------
    if (cfg.modes.count("admixture")) {
        for (auto& pr : cfg.admixPairs) {
            if (pr.a > Traits<T>::kMaxDigit || pr.b > Traits<T>::kMaxDigit) continue;
            char series[32];
            snprintf(series, sizeof series, "%dx%d", pr.a, pr.b);
            printf("  admixture %s (ns/op vs %% of %d-digit):\n", series, pr.a);
            for (int p = 0; p <= 100; p += cfg.admixStep) {
                auto data = GenAdmixture<T>(rng, pr.a, pr.b, p, cfg.size);
                auto res = InterleavedMeasure<T>(fnsNoToy, data, cfg);
                Report<T>(fp, fnsNoToy, res, "admixture", series, p);
                double lo = *std::min_element(res.begin(), res.end());
                printf("    %3d%%  best=%7.3f ns\n", p, lo);
            }
        }

        // 128-bit only: sweep the mix straddling each of zmij's two u128
        // branch thresholds. straddle64 (2^64) exposes the u64<->u128 delegation
        // branch (a boundary in bits); straddle1e32 (1e32) exposes the one-peel
        // (20-32 digits) vs two-peel (33-39) interior-chunk branch. A digit-pair
        // admixture reaches neither.
        if (sizeof(T) > 8) {
            const uint128_t e16 = 10'000'000'000'000'000ull;  // 1e16
            struct { const char* series; uint128_t threshold; } straddles[] = {
                {"straddle64", (uint128_t)1 << 64},
                {"straddle1e32", e16 * e16},  // 1e32
            };
            for (auto& st : straddles) {
                printf("  admixture %s (ns/op vs %% below threshold):\n", st.series);
                for (int p = 0; p <= 100; p += cfg.admixStep) {
                    auto data = GenStraddle<T>(rng, st.threshold, p, cfg.size);
                    auto res = InterleavedMeasure<T>(fnsNoToy, data, cfg);
                    Report<T>(fp, fnsNoToy, res, "admixture", st.series, p);
                    double lo = *std::min_element(res.begin(), res.end());
                    printf("    %3d%% below  best=%7.3f ns\n", p, lo);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// argv parsing
// ---------------------------------------------------------------------------
static std::vector<std::string> Split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t pos = s.find(sep, start);
        if (pos == std::string::npos) { out.push_back(s.substr(start)); break; }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

static void ParseArgs(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto eq = a.find('=');
        std::string key = a.substr(0, eq);
        std::string val = (eq == std::string::npos) ? "" : a.substr(eq + 1);

        if (key == "--no-verify")        cfg.verify = false;
        else if (key == "--modes")       { cfg.modes.clear();  for (auto& m : Split(val, ',')) cfg.modes.insert(m); }
        else if (key == "--types")       { cfg.types.clear();  for (auto& t : Split(val, ',')) cfg.types.insert(t); }
        else if (key == "--filter")      { for (auto& f : Split(val, ',')) if (!f.empty()) cfg.filters.push_back(f); }
        else if (key == "--admix-step")  cfg.admixStep = std::max(1, atoi(val.c_str()));
        else if (key == "--size")        cfg.size = std::max<size_t>(1, strtoull(val.c_str(), nullptr, 10));
        else if (key == "--rounds")      cfg.rounds = std::max(1, atoi(val.c_str()));
        else if (key == "--passes")      cfg.passTarget = std::max<uint64_t>(1, strtoull(val.c_str(), nullptr, 10));
        else if (key == "--out")         cfg.out = val;
        else if (key == "--admix") {
            cfg.admixPairs.clear();
            for (auto& pair : Split(val, ';')) {
                auto ab = Split(pair, ',');
                if (ab.size() == 2) cfg.admixPairs.push_back({ atoi(ab[0].c_str()), atoi(ab[1].c_str()) });
            }
        }
        else if (key == "--uniform") {
            auto lh = Split(val, '-');
            if (lh.size() == 2) {
                cfg.uniformLo = strtoull(lh[0].c_str(), nullptr, 10);
                cfg.uniformHi = strtoull(lh[1].c_str(), nullptr, 10);
            }
        }
        else if (key == "--loguniform") {
            for (auto& w : Split(val, ',')) {
                auto lh = Split(w, '-');
                if (lh.size() == 2) cfg.logWindows.push_back({ atoi(lh[0].c_str()), atoi(lh[1].c_str()) });
            }
        }
        else if (key == "--help" || key == "-h") {
            printf("Usage: itoa [options]\n"
                   "  --modes=bylength,loguniform,unpredictable,admixture,uniform\n"
                   "  --types=u32,i32,u64,i64,u128,i128\n"
                   "  --filter=sse2,jeaiii,fmt        (substring, comma = OR)\n"
                   "  --admix=5,6;9,10                (digit-length pairs)\n"
                   "  --admix-step=10                 (%% step for admixture sweep)\n"
                   "  --loguniform=1-10,1-20          (length windows)\n"
                   "  --uniform=0-255                 (equidistributed value range)\n"
                   "  --size=65536 --rounds=6 --passes=1048576\n"
                   "  --out=results.csv  --no-verify\n");
            exit(0);
        }
        else
            fprintf(stderr, "warning: unknown option '%s'\n", key.c_str());
    }
}

// ---------------------------------------------------------------------------
static FILE* OpenCsv(const Config& cfg) {
    std::string path = cfg.out;
    if (path.empty()) {
        // Prefer ../result if it exists (running from a build dir), else cwd.
        FILE* probe = fopen("../result/template.php", "r");
        if (probe) { fclose(probe); path = "../result/" RESULT_FILENAME; }
        else       { path = RESULT_FILENAME; }
    }
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) { perror(path.c_str()); exit(1); }
    fprintf(fp, "Type,Function,Mode,Series,X,Time_ns\n");
    printf("Writing results to %s\n", path.c_str());
    return fp;
}

int main(int argc, char** argv) {
    Config cfg;
    ParseArgs(argc, argv, cfg);

    TestList& tests = TestManager::Instance().GetTests();
    std::sort(tests.begin(), tests.end(),
              [](const Test* a, const Test* b) { return std::string(a->fname) < std::string(b->fname); });

    if (cfg.verify) VerifyAll(cfg);

    FILE* fp = OpenCsv(cfg);
    RunType<uint32_t>(cfg, fp);
    RunType<int32_t>(cfg, fp);
    RunType<uint64_t>(cfg, fp);
    RunType<int64_t>(cfg, fp);
    RunType<uint128_t>(cfg, fp);
    RunType<int128_t>(cfg, fp);
    fclose(fp);

    return 0;
}
