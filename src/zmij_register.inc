// Shared registration body for a single zmij microarchitecture variant.
//
// Each variant .cpp defines, before including this file:
//   ZMIJ_NS      the (renamed) library namespace, e.g. zmij_avx2
//   ZMIJ_SUFFIX  the benchmark/registration name,   e.g. zmij_avx2
// plus any ZMIJ_USE_* overrides (e.g. ZMIJ_USE_SIMD 0 for the scalar variant),
// and is compiled with the -march level for that tier.
//
// zmij.cc hard-codes `namespace zmij`; we rename it per TU with `#define zmij
// ZMIJ_NS` so all four variants coexist in one binary with distinct symbols.
// The library's public integer API is zmij::detail::itoa<UInt> /
// itoa_signed<Int>, both returning the end pointer WITHOUT a terminator, so the
// wrappers write the '\0'.

#define zmij ZMIJ_NS
#include "zmij.cc"
#undef zmij

#include <cstring>
#include "test.h"

#define ZMIJ_CAT_(a, b) a##b
#define ZMIJ_CAT(a, b) ZMIJ_CAT_(a, b)
#define ZMIJ_FN(base) ZMIJ_CAT(base, ZMIJ_CAT(_, ZMIJ_SUFFIX))

void ZMIJ_FN(u32toa)(uint32_t v, char* b)   { *ZMIJ_NS::detail::itoa(v, b) = '\0'; }
void ZMIJ_FN(i32toa)(int32_t v, char* b)    { *ZMIJ_NS::detail::itoa_signed(v, b) = '\0'; }
#ifdef ZMIJ_U64_FITS_DISPATCH
// Route small u64 values through the cheaper u32 kernel (one 16-byte store,
// single reciprocal group) instead of the u64 two-chunk path. Any threshold
// <= 2^32 is correct (uint32_t(v) == v there). A power-of-10 threshold keeps
// every digit-length class on one side of the branch (no straddle -> no
// per-length misprediction); 2^32 itself sits inside the 10-digit range.
#  ifndef ZMIJ_U64_FITS_THRESHOLD
#    define ZMIJ_U64_FITS_THRESHOLD 4294967296ull  // 2^32
#  endif
void ZMIJ_FN(u64toa)(uint64_t v, char* b) {
  char* e = v < ZMIJ_U64_FITS_THRESHOLD
                ? ZMIJ_NS::detail::itoa(uint32_t(v), b)
                : ZMIJ_NS::detail::itoa(v, b);
  *e = '\0';
}
#else
void ZMIJ_FN(u64toa)(uint64_t v, char* b)   { *ZMIJ_NS::detail::itoa(v, b) = '\0'; }
#endif
void ZMIJ_FN(i64toa)(int64_t v, char* b)    { *ZMIJ_NS::detail::itoa_signed(v, b) = '\0'; }
void ZMIJ_FN(u128toa)(uint128_t v, char* b) { *ZMIJ_NS::detail::itoa(v, b) = '\0'; }
void ZMIJ_FN(i128toa)(int128_t v, char* b)  { *ZMIJ_NS::detail::itoa_signed((__int128)v, b) = '\0'; }

REGISTER_TEST128(ZMIJ_SUFFIX);
