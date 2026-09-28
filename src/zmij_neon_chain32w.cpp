// zmij, same tier as zmij_neon, with the branch-free u32 multiply chain
// taking three digits per step on a 32-bit fraction.
#define ZMIJ_USE_U32_CHAIN 1
#define ZMIJ_U32_CHAIN_DIGITS 3
#define ZMIJ_U32_CHAIN_FRAC32 1
#define ZMIJ_NS zmij_neon_chain32w
#define ZMIJ_SUFFIX zmij_neon_chain32w
#include "zmij_register.inc"
