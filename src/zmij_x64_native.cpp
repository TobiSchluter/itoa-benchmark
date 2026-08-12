// zmij, tuned for the local CPU (-march=native -mtune=native). Same source as
// the other tiers; the compiler uses every ISA extension and scheduling model
// the build host provides. On an AVX2-capable host the 256-bit YMM u128 kernel
// (ZMIJ_USE_AVX2) is on.
#define ZMIJ_NS zmij_x64_native
#define ZMIJ_SUFFIX zmij_x64_native
#include "zmij_register.inc"
