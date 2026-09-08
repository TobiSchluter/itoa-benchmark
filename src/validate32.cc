// Exhaustively cross-checks zmij's integer formatting over the whole 32-bit
// range against a reference writer: jeaiii's itoa by default, fmt with
// --ref=fmt, or zmij itself with --ref=self.
//
// The sweep is always the 32-bit range, but --type= picks which entry point it
// is fed to, so the 64-bit and 128-bit kernels can be swept exhaustively too:
//   i32, i64, i128   itoa_signed over [INT32_MIN, INT32_MAX]
//   u32, u64, u128   itoa over [0, UINT32_MAX]
// Which kernel a value actually reaches inside zmij is an implementation
// detail (the wide entry points delegate to narrower ones for small values),
// so each type is treated as if its own width's kernel ran.
//
// zmij's contract: itoa/itoa_signed return one past the last digit and may
// write anywhere in a buffer of the documented size for the type -- 16 bytes
// for uint32 and 17 for int32, 19/20 for 64-bit, 39/40 for 128-bit. Bytes past
// the returned pointer, up to that size, are unspecified. Both references
// write the string plus a NUL and nothing else.
//
// Because the zmij tail is unspecified, every fixed-width compare first makes
// the buffers a pure function of the value: a zero store of the unsigned size
// (16, 19 or 39 bytes) at the string end covers [len, len + unsigned_size),
// which reaches the last byte of the signed size even for len == 1. On the
// reference side the zero store starts one byte later, at len + 1, so that a
// length disagreement leaves its evidence byte b[len] intact (the reference's
// NUL, or a digit if the reference thinks the number is longer).
//
// Validation strategies, each timed separately:
//   one-by-one len    format one value with both writers, memcmp len+1 bytes
//   one-by-one fixed  zero the tails, compare the signed size (no
//                     length-dependent branches in the compare)
//   batched slots     fill two 256-slot buffers, then compare slot by slot
//                     over len+1 bytes
//   batched memcmp    fill the same buffers with zeroed tails, then compare
//                     both buffers with a single memcmp per batch
//
// The range is split into one chunk per thread, each validated by a
// std::async task; the reported time is the wall-clock time of the whole
// fan-out, so ns/value is aggregate parallel throughput.
//
// Traversal order matters as soon as the timings are read as a statement about
// the writers. Sequential order leaves a branchy writer perfectly predicted --
// across the whole range jeaiii's magnitude branches change ten times and its
// sign branch once -- while zmij is branch-free and gains nothing from it.
// --order=scrambled visits first + ((i * K) & (count - 1)) instead: K is odd,
// so an odd multiply modulo a power-of-two count is a bijection and the sweep
// still covers every value exactly once, in an order no predictor can learn.
// It needs a power-of-two count, which every type's full range is.
//
// Usage: validate32 [--ref=jeaiii|fmt|self] [--type=LIST] [--threads=N]
//                   [--order=sequential|scrambled] [first last]
// LIST is a comma-separated list of i32,u32,i64,u64,i128,u128, or "both" for
// i32,u32 (the default), or "all" for every type. --ref=self validates zmij
// against itself: nothing can fail, so it times the strategies with the
// reference writer's own cost taken out. jeaiii has no 128-bit writer.

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

#include "zmij-int.cc"

using u128 = unsigned __int128;
using i128 = __int128;

void i32toa_jeaiii(int32_t i, char* b);
void u32toa_jeaiii(uint32_t u, char* b);
void i64toa_jeaiii(int64_t i, char* b);
void u64toa_jeaiii(uint64_t u, char* b);
void i32toa_fmt(int32_t i, char* b);
void u32toa_fmt(uint32_t u, char* b);
void i64toa_fmt(int64_t i, char* b);
void u64toa_fmt(uint64_t u, char* b);
void i128toa_fmt(i128 i, char* b);
void u128toa_fmt(u128 u, char* b);

namespace {

// The reference writer: writes the string plus a NUL, like every reference
// above. A non-type template parameter so each runner instantiation calls its
// reference directly.
template <typename T>
using ref_fn = void (*)(T, char*);

// zmij's entry point for each validated type.
auto zmij_write(int32_t v, char* out) -> char* {
  return zmij::details_int::itoa_signed(out, v);
}
auto zmij_write(uint32_t v, char* out) -> char* {
  return zmij::details_int::itoa(out, v);
}
auto zmij_write(int64_t v, char* out) -> char* {
  return zmij::details_int::itoa_signed(out, v);
}
auto zmij_write(uint64_t v, char* out) -> char* {
  return zmij::details_int::itoa(out, v);
}
auto zmij_write(i128 v, char* out) -> char* {
  return zmij::details_int::itoa_signed(out, v);
}
auto zmij_write(u128 v, char* out) -> char* {
  return zmij::details_int::itoa(out, v);
}

// --ref=self: zmij as its own reference. Kept out of line so it stays a call
// the writer under test cannot be folded into, the way the jeaiii and fmt
// references are calls into another translation unit.
template <typename T>
ZMIJ_NOINLINE void self_ref(T v, char* b) {
  *zmij_write(v, b) = '\0';
}

// The two documented buffer sizes for a type of this width: unsigned first,
// then signed (the sign byte shifts the same window one further). Everything
// else is derived from them, so no assumption about which kernel runs is baked
// in anywhere.
template <size_t bytes>
struct widths {
  // The tail zero store: one constant-size store, so the compare has no
  // length-dependent work.
  static constexpr auto zero() -> size_t {
    return bytes == 4 ? 16 : bytes == 8 ? 19 : 39;
  }
  // The fixed compare width. zmij may write any byte below it, and the zero
  // store covers [len, width()) for every len >= 1.
  static constexpr auto width() -> size_t { return zero() + 1; }
  // One slot per value in the batched buffers. Index width() is past
  // everything zmij may write, and past the zero store when len == 1, so it
  // stays zero in both buffers.
  static constexpr auto slot() -> size_t { return width() + 1; }
  // Every sweep feeds 32-bit-range values, so len <= 11 and the last byte any
  // store reaches is max(width(), 11 + zero()).
  static constexpr auto buffer() -> size_t { return bytes == 16 ? 64 : 32; }
};

// Name and inclusive sweep range of each validated type. Ranges are int64_t so
// one signed loop counter covers both signs (uint32's top half included).
template <typename T>
struct domain;
template <>
struct domain<int32_t> : widths<4> {
  static auto name() -> const char* { return "i32"; }
  static constexpr auto first() -> int64_t { return INT32_MIN; }
  static constexpr auto last() -> int64_t { return INT32_MAX; }
};
template <>
struct domain<uint32_t> : widths<4> {
  static auto name() -> const char* { return "u32"; }
  static constexpr auto first() -> int64_t { return 0; }
  static constexpr auto last() -> int64_t { return UINT32_MAX; }
};
template <>
struct domain<int64_t> : widths<8> {
  static auto name() -> const char* { return "i64"; }
  static constexpr auto first() -> int64_t { return INT32_MIN; }
  static constexpr auto last() -> int64_t { return INT32_MAX; }
};
template <>
struct domain<uint64_t> : widths<8> {
  static auto name() -> const char* { return "u64"; }
  static constexpr auto first() -> int64_t { return 0; }
  static constexpr auto last() -> int64_t { return UINT32_MAX; }
};
template <>
struct domain<i128> : widths<16> {
  static auto name() -> const char* { return "i128"; }
  static constexpr auto first() -> int64_t { return INT32_MIN; }
  static constexpr auto last() -> int64_t { return INT32_MAX; }
};
template <>
struct domain<u128> : widths<16> {
  static auto name() -> const char* { return "u128"; }
  static constexpr auto first() -> int64_t { return 0; }
  static constexpr auto last() -> int64_t { return UINT32_MAX; }
};

constexpr size_t batch = 256;

// Odd, so multiplying by it modulo a power of two is a bijection.
constexpr uint32_t scramble_mult = 0x9E3779B1u;

// How a loop index maps to a value. Written once per sweep before the tasks
// are spawned, read-only while they run.
struct {
  int64_t base = 0;   // the value at index 0
  uint32_t mask = 0;  // count - 1, count being a power of two
} order;

// The same map, hoisted into locals: the runners store through char*, which
// may alias anything, so reading `order` inside the loop would reload it.
template <bool scrambled>
struct mapper {
  int64_t base;
  uint32_t mask;
  mapper() : base(order.base), mask(order.mask) {}
  auto operator()(int64_t i) const -> int64_t {
    return scrambled ? base + int64_t((uint32_t(i) * scramble_mult) & mask)
                     : base + i;
  }
};

struct result {
  uint64_t mismatches = 0;
  double seconds = 0;
};

std::atomic<uint64_t> reported{0};

void report(int64_t v, const char* a, const char* b) {
  if (reported.fetch_add(1, std::memory_order_relaxed) < 10)
    fprintf(stderr, "mismatch: %" PRId64 " -> zmij \"%s\" reference \"%s\"\n",
            v, a, b);
}

using clock_type = std::chrono::steady_clock;

auto since(clock_type::time_point start) -> double {
  return std::chrono::duration<double>(clock_type::now() - start).count();
}

template <typename T, ref_fn<T> Ref, bool scrambled>
auto run_single(int64_t lo, int64_t hi) -> uint64_t {
  const mapper<scrambled> at;
  uint64_t mismatches = 0;
  for (int64_t i = lo; i <= hi; ++i) {
    int64_t v = at(i);
    char a[domain<T>::buffer()], b[domain<T>::buffer()];
    char* end = zmij_write(T(v), a);
    *end = '\0';
    Ref(T(v), b);
    if (memcmp(a, b, size_t(end - a) + 1) != 0) {
      ++mismatches;
      report(v, a, b);
    }
  }
  return mismatches;
}

template <typename T, ref_fn<T> Ref, bool scrambled>
auto run_single_fixed(int64_t lo, int64_t hi) -> uint64_t {
  const mapper<scrambled> at;
  uint64_t mismatches = 0;
  for (int64_t i = lo; i <= hi; ++i) {
    int64_t v = at(i);
    char a[domain<T>::buffer()], b[domain<T>::buffer()];
    char* end = zmij_write(T(v), a);
    memset(end, 0, domain<T>::zero());
    Ref(T(v), b);
    memset(b + (end - a) + 1, 0, domain<T>::zero());
    if (memcmp(a, b, domain<T>::width()) != 0) {
      ++mismatches;
      report(v, a, b);
    }
  }
  return mismatches;
}

// bulk == false: NUL-terminate and compare lens[i] + 1 bytes per slot.
//
// bulk == true: zero the slot tails as in run_single_fixed so one memcmp
// validates the whole batch. The zero stores spill into the next slot, which
// that slot's own writes then cover, and each slot's last index is never
// written by anything, so it stays zero from the initial memset.
template <typename T, ref_fn<T> Ref, bool bulk, bool scrambled>
auto run_batched(int64_t lo, int64_t hi) -> uint64_t {
  constexpr size_t slot = domain<T>::slot();
  const mapper<scrambled> at;
  uint64_t mismatches = 0;
  // The buffers get 16 bytes of padding because the tail-zeroing stores of
  // the last slot may run past it. Task-local so concurrent chunks don't
  // share them.
  alignas(64) char buf_a[slot * batch + 16];
  alignas(64) char buf_b[slot * batch + 16];
  memset(buf_a, 0, sizeof buf_a);
  memset(buf_b, 0, sizeof buf_b);
  uint8_t lens[batch];
  for (int64_t start = lo; start <= hi; start += batch) {
    size_t n = hi - start + 1 < int64_t(batch) ? size_t(hi - start + 1) : batch;
    char* pa = buf_a;
    for (size_t i = 0; i < n; ++i, pa += slot) {
      char* end = zmij_write(T(at(start + int64_t(i))), pa);
      lens[i] = uint8_t(end - pa);
      if (bulk)
        memset(end, 0, domain<T>::zero());
      else
        *end = '\0';
    }
    char* pb = buf_b;
    for (size_t i = 0; i < n; ++i, pb += slot) {
      Ref(T(at(start + int64_t(i))), pb);
      if (bulk) memset(pb + lens[i] + 1, 0, domain<T>::zero());
    }
    if (bulk && memcmp(buf_a, buf_b, slot * n) == 0) continue;
    for (size_t i = 0; i < n; ++i) {
      const char* a = buf_a + i * slot;
      const char* b = buf_b + i * slot;
      if (memcmp(a, b, size_t(lens[i]) + 1) != 0) {
        ++mismatches;
        report(at(start + int64_t(i)), a, b);
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

template <typename T, ref_fn<T> Ref>
auto run_all(const char* ref_name, int64_t first, int64_t last,
             unsigned threads, bool scrambled) -> uint64_t {
  uint64_t count = uint64_t(last - first) + 1;
  if (scrambled && (count & (count - 1)) != 0) {
    printf("skipping %s: --order=scrambled needs a power-of-two count, got %"
           PRIu64 "\n",
           domain<T>::name(), count);
    return 0;
  }
  order.base = first;
  order.mask = uint32_t(count - 1);
  printf("validating zmij %s against %s over [%" PRId64 ", %" PRId64
         "], %u threads, %s order\n",
         domain<T>::name(), ref_name, first, last, threads,
         scrambled ? "scrambled" : "sequential");
  fflush(stdout);
  uint64_t total = 0;
  struct strategy {
    const char* name;
    uint64_t (*fn)(int64_t, int64_t);
  };
  const strategy in_order[] = {
      {"one-by-one len", run_single<T, Ref, false>},
      {"one-by-one fixed", run_single_fixed<T, Ref, false>},
      {"batched slots", run_batched<T, Ref, false, false>},
      {"batched memcmp", run_batched<T, Ref, true, false>},
  };
  const strategy shuffled[] = {
      {"one-by-one len", run_single<T, Ref, true>},
      {"one-by-one fixed", run_single_fixed<T, Ref, true>},
      {"batched slots", run_batched<T, Ref, false, true>},
      {"batched memcmp", run_batched<T, Ref, true, true>},
  };
  const strategy* strategies = scrambled ? shuffled : in_order;
  // The runners take index bounds, not values; the map above turns them back.
  int64_t top = int64_t(count) - 1;
  for (int k = 0; k < 4; ++k) {
    result r = run_parallel(strategies[k].fn, 0, top, threads);
    print(strategies[k].name, r, 0, top);
    total += r.mismatches;
  }
  return total;
}

// Intersects the requested range with T's sweep range. Returns false, after
// saying so, when the intersection is empty.
template <typename T>
auto select_range(bool have_range, int64_t first, int64_t last, int64_t* lo,
                  int64_t* hi) -> bool {
  *lo = have_range && first > domain<T>::first() ? first : domain<T>::first();
  *hi = have_range && last < domain<T>::last() ? last : domain<T>::last();
  if (*lo > *hi) {
    printf("skipping %s: [%" PRId64 ", %" PRId64 "] is outside its range\n",
           domain<T>::name(), first, last);
    return false;
  }
  return true;
}

// Maps --ref= to this type's reference writer.
template <typename T, ref_fn<T> Jeaiii, ref_fn<T> Fmt>
auto run_refs(const char* ref, int64_t lo, int64_t hi, unsigned threads,
              bool scrambled) -> uint64_t {
  if (strcmp(ref, "self") == 0)
    return run_all<T, self_ref<T> >(ref, lo, hi, threads, scrambled);
  if (strcmp(ref, "fmt") == 0)
    return run_all<T, Fmt>(ref, lo, hi, threads, scrambled);
  return run_all<T, Jeaiii>(ref, lo, hi, threads, scrambled);
}

// Same for the 128-bit types, which jeaiii has no writer for.
template <typename T, ref_fn<T> Fmt>
auto run_refs_without_jeaiii(const char* ref, int64_t lo, int64_t hi,
                             unsigned threads, bool scrambled) -> uint64_t {
  if (strcmp(ref, "jeaiii") == 0) {
    printf("skipping %s: jeaiii has no 128-bit writer (use --ref=fmt or "
           "--ref=self)\n",
           domain<T>::name());
    return 0;
  }
  if (strcmp(ref, "self") == 0)
    return run_all<T, self_ref<T> >(ref, lo, hi, threads, scrambled);
  return run_all<T, Fmt>(ref, lo, hi, threads, scrambled);
}

const char* const type_names[] = {"i32", "u32", "i64", "u64", "i128", "u128"};

// --type= is a comma-separated list of type names, "both" for i32,u32, or
// "all" for every type.
auto selected(const char* list, const char* name) -> bool {
  if (strcmp(list, "all") == 0) return true;
  if (strcmp(list, "both") == 0)
    return strcmp(name, "i32") == 0 || strcmp(name, "u32") == 0;
  size_t want = strlen(name);
  for (const char* p = list; *p;) {
    const char* comma = strchr(p, ',');
    size_t n = comma ? size_t(comma - p) : strlen(p);
    if (n == want && strncmp(p, name, n) == 0) return true;
    p = comma ? comma + 1 : p + n;
  }
  return false;
}

// Rejects a typo in --type= instead of silently narrowing the run.
auto types_known(const char* list) -> bool {
  if (strcmp(list, "all") == 0 || strcmp(list, "both") == 0) return true;
  for (const char* p = list; *p;) {
    const char* comma = strchr(p, ',');
    size_t n = comma ? size_t(comma - p) : strlen(p);
    bool found = false;
    for (const char* known : type_names)
      if (n == strlen(known) && strncmp(p, known, n) == 0) found = true;
    if (!found) return false;
    p = comma ? comma + 1 : p + n;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* ref = "jeaiii";
  const char* type = "both";
  const char* traversal = "sequential";
  unsigned threads = std::thread::hardware_concurrency();
  if (threads == 0) threads = 1;
  int64_t range[2];
  int nrange = 0;
  bool ok = true;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--ref=", 6) == 0)
      ref = argv[i] + 6;
    else if (strncmp(argv[i], "--type=", 7) == 0)
      type = argv[i] + 7;
    else if (strncmp(argv[i], "--order=", 8) == 0)
      traversal = argv[i] + 8;
    else if (strncmp(argv[i], "--threads=", 10) == 0)
      threads = unsigned(atoi(argv[i] + 10));
    else if (nrange < 2)
      range[nrange++] = atoll(argv[i]);
    else
      ok = false;
  }
  if (threads == 0) ok = false;
  bool have_range = nrange == 2;
  int64_t first = INT32_MIN, last = UINT32_MAX;
  if (have_range) {
    first = range[0];
    last = range[1];
  } else if (nrange != 0) {
    ok = false;
  }
  if (first < INT32_MIN || last > UINT32_MAX || first > last) {
    fprintf(stderr, "invalid range [%" PRId64 ", %" PRId64 "]\n", first, last);
    return 2;
  }
  if (strcmp(ref, "jeaiii") != 0 && strcmp(ref, "fmt") != 0 &&
      strcmp(ref, "self") != 0)
    ok = false;
  bool scrambled = strcmp(traversal, "scrambled") == 0;
  if (!scrambled && strcmp(traversal, "sequential") != 0) ok = false;
  if (!types_known(type)) ok = false;
  if (!ok) {
    fprintf(stderr,
            "usage: %s [--ref=jeaiii|fmt|self] [--type=LIST] [--threads=N] "
            "[--order=sequential|scrambled] [first last]\n"
            "  LIST: comma-separated i32,u32,i64,u64,i128,u128 -- or both "
            "(i32,u32) or all\n"
            "  scrambled order needs a power-of-two count, which every full "
            "range is\n",
            argv[0]);
    return 2;
  }
  uint64_t mismatches = 0;
  int64_t lo, hi;
  if (selected(type, "i32") &&
      select_range<int32_t>(have_range, first, last, &lo, &hi))
    mismatches += run_refs<int32_t, i32toa_jeaiii, i32toa_fmt>(
        ref, lo, hi, threads, scrambled);
  if (selected(type, "u32") &&
      select_range<uint32_t>(have_range, first, last, &lo, &hi))
    mismatches += run_refs<uint32_t, u32toa_jeaiii, u32toa_fmt>(
        ref, lo, hi, threads, scrambled);
  if (selected(type, "i64") &&
      select_range<int64_t>(have_range, first, last, &lo, &hi))
    mismatches += run_refs<int64_t, i64toa_jeaiii, i64toa_fmt>(
        ref, lo, hi, threads, scrambled);
  if (selected(type, "u64") &&
      select_range<uint64_t>(have_range, first, last, &lo, &hi))
    mismatches += run_refs<uint64_t, u64toa_jeaiii, u64toa_fmt>(
        ref, lo, hi, threads, scrambled);
  if (selected(type, "i128") &&
      select_range<i128>(have_range, first, last, &lo, &hi))
    mismatches += run_refs_without_jeaiii<i128, i128toa_fmt>(ref, lo, hi,
                                                             threads,
                                                             scrambled);
  if (selected(type, "u128") &&
      select_range<u128>(have_range, first, last, &lo, &hi))
    mismatches += run_refs_without_jeaiii<u128, u128toa_fmt>(ref, lo, hi,
                                                             threads,
                                                             scrambled);
  return mismatches != 0 ? 1 : 0;
}
