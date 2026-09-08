// Shared registration body for a single zmij microarchitecture variant.
//
// Each variant .cpp defines, before including this file:
//   ZMIJ_NS      the (renamed) library namespace, e.g. zmij_avx2
//   ZMIJ_SUFFIX  the benchmark/registration name,   e.g. zmij_avx2
// plus any ZMIJ_USE_* overrides (e.g. ZMIJ_USE_SIMD 0 for the scalar variant),
// and is compiled with the -march level for that tier.
//
// zmij-int.cc hard-codes `namespace zmij`; we rename it per TU with `#define
// zmij ZMIJ_NS` so all variants coexist in one binary with distinct symbols.
// The library's public integer API is zmij::details_int::itoa<UInt> /
// itoa_signed<Int>, both returning the end pointer WITHOUT a terminator, so the
// wrappers write the '\0'.

#define zmij ZMIJ_NS
#include "zmij-int.cc"
#undef zmij

#include <cstring>
#include "test.h"

#define ZMIJ_CAT_(a, b) a##b
#define ZMIJ_CAT(a, b) ZMIJ_CAT_(a, b)
#define ZMIJ_FN(base) ZMIJ_CAT(base, ZMIJ_CAT(_, ZMIJ_SUFFIX))

void ZMIJ_FN(u32toa)(uint32_t v, char* b)   { *ZMIJ_NS::details_int::itoa(b, v) = '\0'; }
void ZMIJ_FN(i32toa)(int32_t v, char* b)    { *ZMIJ_NS::details_int::itoa_signed(b, v) = '\0'; }
void ZMIJ_FN(u64toa)(uint64_t v, char* b)   { *ZMIJ_NS::details_int::itoa(b, v) = '\0'; }
void ZMIJ_FN(i64toa)(int64_t v, char* b)    { *ZMIJ_NS::details_int::itoa_signed(b, v) = '\0'; }
void ZMIJ_FN(u128toa)(uint128_t v, char* b) { *ZMIJ_NS::details_int::itoa(b, (unsigned __int128)v) = '\0'; }
void ZMIJ_FN(i128toa)(int128_t v, char* b)  { *ZMIJ_NS::details_int::itoa_signed(b, (__int128)v) = '\0'; }

REGISTER_TEST128(ZMIJ_SUFFIX);
