# itoa Benchmark

Copyright(c) 2014-2016 Milo Yip (miloyip@gmail.com)

## Introduction

This benchmark evaluates the performance of conversion from integer to ASCII string in decimal. The function prototypes are:

~~~~~~~~cpp
void u32toa(uint32_t value, char* buffer);
void i32toa(int32_t value, char* buffer);
void u64toa(uint64_t value, char* buffer);
void i64toa(int64_t value, char* buffer);
void u128toa(unsigned __int128 value, char* buffer);   // opt-in (fmt/naive)
void i128toa(__int128 value, char* buffer);            // opt-in (fmt/naive)
~~~~~~~~

Note that `itoa()` is *not* a standard function in C and C++, but provided by some compilers.

This fork extends the original benchmark with:

* **128-bit support** (`__int128`). There is no standard formatter for it, so
  the [{fmt}](https://github.com/fmtlib/fmt) library is used
  (`fmt::format_to(buf, "{}", value)`). Most implementations do not provide
  128-bit conversion and are skipped for that width.
* **{fmt} in the roster**: `fmt::format_int` for 32/64-bit, `fmt::format_to`
  for 128-bit.
* **Signed and unsigned in parallel** for every width (`u32/i32`, `u64/i64`,
  `u128/i128`), so the cost of the sign branch is directly comparable.
* **Richer digit distributions** beyond a single fixed length (see below).
* **ABBA / interleaved measurement** to cancel monotonic drift (turbo ramp,
  thermal): within each round every implementation is timed once over the same
  dataset and the roster order is reversed every other round, so each function
  is sampled in both early and late slots; the reported figure is the minimum
  over rounds.
* **Plots** generated from the result CSV with matplotlib.

## Procedure

Firstly the program verifies the correctness of implementations against `naive`.

Then the following benchmark modes are carried out (each over a dataset of
random values with a controlled *digit-length distribution*, converted many
times with the ABBA/interleaved engine):

1. **bylength** — all values have exactly *D* digits, swept over every *D*.
   This is the predictable / best case (the predictor learns the constant
   length). Plotted as ns/op vs digit count.

2. **loguniform** — a shuffled set with an equal number of values in each
   digit-length bin, over a window of lengths (or all lengths). "Log-uniform"
   means uniform over the number of digits, i.e. uniform in `log10(value)`.
   The full-range window is the classic "random" case.

3. **unpredictable** — a [dtolnay](https://github.com/dtolnay/itoa-benchmark)-style
   estimator of the marginal cost of a length-*D* value while the branch
   predictor is thrashed by surrounding random-length noise: measure
   `noise ++ (values of length D)` and subtract the noise-only baseline. Because
   half the data is fixed noise and two minima are subtracted, this is a noisy
   estimator — kept for comparability.

4. **admixture** — a shuffled mix of just **two** digit lengths (default 5 and
   6) at a controlled ratio (0%, 10%, …, 100%). Sweeping the ratio isolates the
   cost of **branch misprediction** on the digit-count branch: the pure ends
   predict perfectly, a ~50/50 mix maximises mispredicts and forms a hump. The
   pair must straddle an implementation's digit-count boundary to expose its
   branch cost.

Every mode runs for all requested widths, signed and unsigned.

## Build and Run

Requirements: CMake ≥ 3.20, a C++17 compiler, and network access on first
configure (CMake fetches {fmt} via `FetchContent`). Presets are provided for
`g++-16` (default) and `clang++-21`.

~~~~~~~~sh
make                       # build + run + plot with g++-16
make PRESET=clang21        # same, with clang++-21
make plots                 # regenerate PNGs from the CSV
~~~~~~~~

Or drive CMake directly:

~~~~~~~~sh
cmake --preset gcc16
cmake --build build/gcc16 -j
./build/gcc16/itoa --out=result/gcc16.csv       # see --help for options
.venv/bin/python plot_results.py result/gcc16.csv
~~~~~~~~

Useful driver options (`itoa --help`):

~~~~~~~~
--modes=bylength,loguniform,unpredictable,admixture
--types=u32,i32,u64,i64,u128,i128
--filter=sse2,jeaiii,fmt        substring match (comma = OR)
--admix=5,6;9,10                digit-length pairs for the admixture sweep
--admix-step=10                 percentage step of the sweep
--loguniform=1-10,1-20          length windows
--size=4096 --rounds=6 --passes=1048576
--out=results.csv  --no-verify
~~~~~~~~

Results are written as a CSV (`Type,Function,Mode,Series,X,Time_ns`) and
`plot_results.py` renders one PNG per (mode, width class) into `result/plots`,
with signed/unsigned drawn side by side.

### Legacy premake build

The original [premake5](http://industriousone.com/premake/download) build has
been superseded by CMake (needed to pull in {fmt}); the `build/` premake files
are kept for reference.

## Results

The following are `sequential` results measured on a PC (Core i7 920 @2.67Ghz), where `u32toa()` is compiled by Visual C++ 2013 and run on Windows 64-bit. The speedup is based on `sprintf()`.

|Function |Time (ns)|Speedup|
|---------|--------:|------:|
|sprintf  |  194.225|  1.00x|
|vc       |   61.522|  3.16x|
|naive    |   26.743|  7.26x|
|count    |   20.552|  9.45x|
|lut      |   17.810| 10.91x|
|countlut |    9.926| 19.57x|
|branchlut|    8.430| 23.04x|
|sse2     |    7.614| 25.51x|
|null     |    2.230| 87.09x|

![corei7920@2.67_win64_vc2013_u32toa_sequential_time](result/corei7920@2.67_win64_vc2013_u32toa_sequential_time.png)

![corei7920@2.67_win64_vc2013_u32toa_sequential_timedigit](result/corei7920@2.67_win64_vc2013_u32toa_sequential_timedigit.png)

Note that the `null` implementation does nothing. It measures the overheads of looping and function call.

Since the C++ standard library implementations (`ostringstream`, `ostrstream`, `to_string`) are slow, they are turned off by default. User can re-enable them by defining `RUN_CPPITOA` macro.

Some results of various configurations are located at `itoa-benchmark/result`. They can be accessed online, with interactivity provided by [Google Charts](https://developers.google.com/chart/):

* [corei7920@2.67_win32_vc2013](http://rawgit.com/miloyip/itoa-benchmark/master/result/corei7920@2.67_win32_vc2013.html)
* [corei7920@2.67_win64_vc2013](http://rawgit.com/miloyip/itoa-benchmark/master/result/corei7920@2.67_win64_vc2013.html)
* [corei7920@2.67_cygwin32_gcc4.8](http://rawgit.com/miloyip/itoa-benchmark/master/result/corei7920@2.67_cygwin32_gcc4.8.html)
* [corei7920@2.67_cygwin64_gcc4.8](http://rawgit.com/miloyip/itoa-benchmark/master/result/corei7920@2.67_cygwin64_gcc4.8.html)

## Implementations

Function      | Description
--------------|-----------
ostringstream | `std::ostringstream` in C++ standard library.
ostrstream    | `std::ostrstream` in C++ standard library.
to_string     | `std::to_string()` in C++11 standard library.
sprintf       | `sprintf()` in C standard library
vc            | Visual C++'s `_itoa()`, `_i64toa()`, `_ui64toa()`
naive         | Compute division/modulo of 10 for each digit, store digits in temp array and copy to buffer in reverse order.
unnamed       | Compute division/modulo of 10 for each digit, store directly in buffer
count         | Count number of decimal digits first, using technique from [1].
lut           | Uses lookup table (LUT) of digit pairs for division/modulo of 100. Mentioned in [2]
countlut      | Combines count and lut.
branchlut     | Use branching to divide-and-conquer the range of value, make computation more parallel.
sse2          | Based on branchlut scheme, use SSE2 SIMD instructions to convert 8 digits in parallel. The algorithm is designed by Wojciech Muła [3]. (Experiment shows it is useful for values equal to or more than 9 digits)
fmt           | [{fmt}](https://github.com/fmtlib/fmt): `fmt::format_int` for 32/64-bit and `fmt::format_to(buf, "{}", value)` for 128-bit.
null          | Do nothing.

## Warm-up and the AVX↔SSE transition penalty

Two guards against measurement distortion:

* **Warm-up.** Before the timed rounds, every implementation is run once over
  the dataset untimed (primes I/D caches and the branch predictor, lets the core
  ramp to its turbo frequency). The ABBA min-over-rounds then discards any
  remaining cold sample.
* **`vzeroupper` between timed blocks.** An AVX routine that leaves the upper
  128 bits of the YMM registers "dirty" makes subsequent *legacy*-SSE code pay a
  per-instruction transition penalty until the state is cleared. Since the engine
  times different implementations back to back (an AVX one right before a
  legacy-SSE one), the driver issues `_mm256_zeroupper()` before each timed block
  so that penalty is not misattributed to the following implementation.

## FAQ

1. How to add an implementation?
   
   You may clone an existing implementation file (e.g. `naive.cpp`). And then modify it. Re-run `premake` to add it to project or makefile. Note that it will automatically register to the benchmark by macro `REGISTER_TEST(name)`.

   Making pull request of new implementations is welcome.

2. Why not converting integers to `std::string`?

   It may introduce heap allocation, which is a big overhead. User can easily wrap these low-level functions to return `std::string`, if needed.

3. Why fast `itoa()` functions is needed?

   They are a very common operations in writing data in text format. The standard way of `sprintf()`, `std::stringstream`, `std::to_string(int)` (C++11) often provides poor performance. The author of this benchmark would optimize the "naive" implementation in [RapidJSON](https://github.com/miloyip/rapidjson/issues/31), thus he creates this project.

## References

[1] Anderson, [Bit Twiddling Hacks](https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog10), 1997.

[2] Alexandrescu, [Three Optimization Tips for C++](http://www.slideshare.net/andreialexandrescu1/three-optimization-tips-for-c-15708507), 2012.

[3] Muła, [SSE: conversion integers to decimal representation](http://wm.ite.pl/articles/sse-itoa.html), 2011.

## Related Benchmarks and Discussions

* [The String Formatters of Manor Farm] (http://www.gotw.ca/publications/mill19.htm) by Herb Sutter, 2001.
* [C++ itoa benchmark](https://github.com/localvoid/cxx-benchmark-itoa) by [localvoid](https://github.com/localvoid)
* [Stackoverflow: C++ performance challenge: integer to std::string conversion](http://stackoverflow.com/questions/4351371/c-performance-challenge-integer-to-stdstring-conversion)
