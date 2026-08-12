// zmij, NEON path compiled for x86 through the neon2sse shim. Lets the NEON
// kernels be benchmarked next to the x64 tiers on a single machine; see
// src/neon2sse/arm_neon.h for what the numbers do and do not mean. Built only
// when ITOA_NEON2SSE=ON, which supplies the include paths and the
// ZMIJ_USE_NEON / ZMIJ_USE_SSE / ZMIJ_NEON2SSE_SHIM definitions.
#define ZMIJ_NS zmij_neon_shim
#define ZMIJ_SUFFIX zmij_neon_shim
#include "zmij_register.inc"
