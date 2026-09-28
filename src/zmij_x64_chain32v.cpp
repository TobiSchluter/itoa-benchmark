// zmij, same tier as zmij_x64_native, with the branch-free u32 multiply chain
// taking two digits per step on a 32-bit fraction.
#define ZMIJ_USE_U32_CHAIN 1
#define ZMIJ_U32_CHAIN_FRAC32 1
#define ZMIJ_NS zmij_x64_chain32v
#define ZMIJ_SUFFIX zmij_x64_chain32v
#include "zmij_register.inc"
