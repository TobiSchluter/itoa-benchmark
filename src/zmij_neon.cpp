// zmij, NEON variant (AArch64). NEON is auto-enabled via __ARM_NEON when
// compiled for a native ARM target. This is the SIMD counterpart to the x86
// sse41/avx2 tiers on Apple Silicon and other AArch64 CPUs.
#define ZMIJ_NS zmij_neon
#define ZMIJ_SUFFIX zmij_neon
#include "zmij_register.inc"
