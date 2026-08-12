// zmij, x86-64-v4 (AVX-512F/BW/DQ/VL on top of v3). The source uses fixed
// SSE/AVX2 intrinsics, so v4 mainly changes codegen (EVEX encodings, 32 vector
// regs, Zen5/AVX-512 scheduling); the 256-bit YMM u128 kernel (ZMIJ_USE_AVX2)
// stays on. Compiled at -march=x86-64-v4.
#define ZMIJ_NS zmij_x64_v4
#define ZMIJ_SUFFIX zmij_x64_v4
#include "zmij_register.inc"
