// zmij, AVX2 variant (x86-64-v3: AVX2/BMI/FMA). Enables the 256-bit ymm path
// for the 33-39 digit u128 case. Compiled at -march=x86-64-v3.
#define ZMIJ_NS zmij_avx2
#define ZMIJ_SUFFIX zmij_avx2
#include "zmij_register.inc"
