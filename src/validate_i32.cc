// Exhaustively cross-checks zmij's int32 formatting against a reference
// writer: jeaiii's itoa by default, or fmt with --ref=fmt.
//
// zmij's contract: itoa_signed needs a 17-byte buffer for int32 and returns
// one past the last digit; bytes past that, up to the buffer size, are
// unspecified. Both references write the string plus a NUL and nothing else.
//
// Because the zmij tail is unspecified, every fixed-width compare first makes
// the buffers a pure function of the value: a 16-byte zero store at the
// string end covers [len, len+15], which reaches index 16 even for len == 1.
// On the reference side the zero store starts one byte later, at len + 1, so
// that a length disagreement leaves its evidence byte b[len] intact (the
// reference's NUL, or a digit if the reference thinks the number is longer).
//
// Validation strategies, each timed separately:
//   one-by-one len    format one value with both writers, memcmp len+1 bytes
//   one-by-one fixed  zero the tails, compare 17 bytes (no length-dependent
//                     branches in the compare)
//   batched slots     fill two 256-slot buffers (18 bytes per slot), then
//                     compare slot by slot over len+1 bytes
//   batched memcmp    fill the same buffers with zeroed tails, then compare
//                     both buffers with a single memcmp per batch
//
// The range is split into one chunk per thread, each validated by a
// std::async task; the reported time is the wall-clock time of the whole
// fan-out, so ns/value is aggregate parallel throughput.
//
// Usage: validate_i32 [--ref=jeaiii|fmt] [--threads=N] [first last]
// (defaults to jeaiii, one thread per hardware thread, all of int32)

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

#include "zmij.cc"

void i32toa_jeaiii(int32_t i, char* b);
void i32toa_fmt(int32_t i, char* b);

namespace {

// The reference writer: writes the string plus a NUL, like both references
// above. A non-type template parameter so each runner instantiation calls its
// reference directly.
using ref_fn = void (*)(int32_t, char*);

constexpr size_t slot = 18;
constexpr size_t batch = 256;

struct result {
  uint64_t mismatches = 0;
  double seconds = 0;
};

std::atomic<uint64_t> reported{0};

void report(int32_t v, const char* a, const char* b) {
  if (reported.fetch_add(1, std::memory_order_relaxed) < 10)
    fprintf(stderr, "mismatch: %" PRId32 " -> zmij \"%s\" reference \"%s\"\n",
            v, a, b);
}

using clock_type = std::chrono::steady_clock;

auto since(clock_type::time_point start) -> double {
  return std::chrono::duration<double>(clock_type::now() - start).count();
}

template <ref_fn Ref>
auto run_single(int64_t first, int64_t last) -> uint64_t {
  uint64_t mismatches = 0;
  for (int64_t v = first; v <= last; ++v) {
    char a[32], b[32];
    char* end = zmij::detail::itoa_signed(int32_t(v), a);
    *end = '\0';
    Ref(int32_t(v), b);
    if (memcmp(a, b, size_t(end - a) + 1) != 0) {
      ++mismatches;
      report(int32_t(v), a, b);
    }
  }
  return mismatches;
}

template <ref_fn Ref>
auto run_single_fixed(int64_t first, int64_t last) -> uint64_t {
  uint64_t mismatches = 0;
  for (int64_t v = first; v <= last; ++v) {
    // 32 bytes so the zero stores (at most at offset 11 + 1 + 16) fit.
    char a[32], b[32];
    char* end = zmij::detail::itoa_signed(int32_t(v), a);
    memset(end, 0, 16);
    Ref(int32_t(v), b);
    memset(b + (end - a) + 1, 0, 16);
    if (memcmp(a, b, 17) != 0) {
      ++mismatches;
      report(int32_t(v), a, b);
    }
  }
  return mismatches;
}

// bulk == false: NUL-terminate and compare lens[i] + 1 bytes per slot.
//
// bulk == true: zero the slot tails as in run_single_fixed so one memcmp
// validates the whole batch. The zero stores spill up to 9 bytes into the
// next slot, which the next slot's own writes then cover, and slot index 17
// is never written by anything, so it stays zero from the initial memset.
template <ref_fn Ref, bool bulk>
auto run_batched(int64_t first, int64_t last) -> uint64_t {
  uint64_t mismatches = 0;
  // The buffers get 16 bytes of padding because the tail-zeroing stores of
  // the last slot may run past it. Task-local so concurrent chunks don't
  // share them.
  alignas(64) char buf_a[slot * batch + 16];
  alignas(64) char buf_b[slot * batch + 16];
  memset(buf_a, 0, sizeof buf_a);
  memset(buf_b, 0, sizeof buf_b);
  uint8_t lens[batch];
  for (int64_t base = first; base <= last; base += batch) {
    size_t n = last - base + 1 < int64_t(batch) ? size_t(last - base + 1)
                                                : batch;
    char* pa = buf_a;
    for (size_t i = 0; i < n; ++i, pa += slot) {
      char* end = zmij::detail::itoa_signed(int32_t(base + int64_t(i)), pa);
      lens[i] = uint8_t(end - pa);
      if (bulk)
        memset(end, 0, 16);
      else
        *end = '\0';
    }
    char* pb = buf_b;
    for (size_t i = 0; i < n; ++i, pb += slot) {
      Ref(int32_t(base + int64_t(i)), pb);
      if (bulk) memset(pb + lens[i] + 1, 0, 16);
    }
    if (bulk && memcmp(buf_a, buf_b, slot * n) == 0) continue;
    for (size_t i = 0; i < n; ++i) {
      const char* a = buf_a + i * slot;
      const char* b = buf_b + i * slot;
      if (memcmp(a, b, size_t(lens[i]) + 1) != 0) {
        ++mismatches;
        report(int32_t(base + int64_t(i)), a, b);
      }
    }
  }
  return mismatches;
}

void print(const char* name, result r, int64_t first, int64_t last) {
  double count = double(last - first) + 1;
  printf("%-16s %7.2f s  (%.3f ns/value)  %" PRIu64 " mismatches\n", name,
         r.seconds, r.seconds / count * 1e9, r.mismatches);
  fflush(stdout);
}

// Splits [first, last] into one chunk per thread and validates the chunks
// concurrently. Chunk boundaries are aligned to the batch size so only each
// chunk's final batch can be partial.
auto run_parallel(uint64_t (*fn)(int64_t, int64_t), int64_t first,
                  int64_t last, unsigned threads) -> result {
  result r;
  uint64_t count = uint64_t(last - first) + 1;
  uint64_t chunk = (count / threads + batch) / batch * batch;
  auto start = clock_type::now();
  std::vector<std::future<uint64_t>> futures;
  for (int64_t lo = first; lo <= last; lo += chunk) {
    int64_t hi = last - lo < int64_t(chunk) ? last : lo + int64_t(chunk) - 1;
    futures.push_back(std::async(std::launch::async, fn, lo, hi));
  }
  for (auto& f : futures) r.mismatches += f.get();
  r.seconds = since(start);
  return r;
}

template <ref_fn Ref>
auto run_all(const char* ref_name, int64_t first, int64_t last,
             unsigned threads) -> uint64_t {
  printf("validating zmij against %s over [%" PRId64 ", %" PRId64
         "], %u threads\n",
         ref_name, first, last, threads);
  fflush(stdout);
  uint64_t total = 0;
  struct strategy {
    const char* name;
    uint64_t (*fn)(int64_t, int64_t);
  };
  const strategy strategies[] = {
      {"one-by-one len", run_single<Ref>},
      {"one-by-one fixed", run_single_fixed<Ref>},
      {"batched slots", run_batched<Ref, false>},
      {"batched memcmp", run_batched<Ref, true>},
  };
  for (const auto& s : strategies) {
    result r = run_parallel(s.fn, first, last, threads);
    print(s.name, r, first, last);
    total += r.mismatches;
  }
  return total;
}

}  // namespace

int main(int argc, char** argv) {
  int64_t first = INT32_MIN, last = INT32_MAX;
  const char* ref = "jeaiii";
  unsigned threads = std::thread::hardware_concurrency();
  if (threads == 0) threads = 1;
  int64_t range[2];
  int nrange = 0;
  bool ok = true;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--ref=", 6) == 0)
      ref = argv[i] + 6;
    else if (strncmp(argv[i], "--threads=", 10) == 0)
      threads = unsigned(atoi(argv[i] + 10));
    else if (nrange < 2)
      range[nrange++] = atoll(argv[i]);
    else
      ok = false;
  }
  if (threads == 0) ok = false;
  if (nrange == 2) {
    first = range[0];
    last = range[1];
  } else if (nrange != 0) {
    ok = false;
  }
  if (first < INT32_MIN || last > INT32_MAX || first > last) {
    fprintf(stderr, "invalid range [%" PRId64 ", %" PRId64 "]\n", first, last);
    return 2;
  }
  bool jeaiii = strcmp(ref, "jeaiii") == 0;
  if (!ok || (!jeaiii && strcmp(ref, "fmt") != 0)) {
    fprintf(stderr, "usage: %s [--ref=jeaiii|fmt] [--threads=N] [first last]\n",
            argv[0]);
    return 2;
  }
  uint64_t mismatches = jeaiii
                            ? run_all<i32toa_jeaiii>(ref, first, last, threads)
                            : run_all<i32toa_fmt>(ref, first, last, threads);
  return mismatches != 0 ? 1 : 0;
}
