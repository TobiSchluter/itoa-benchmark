// zmij, same tier as zmij_x64_native, with the branch-free u32 multiply chain
// taking three digits per step.
#define ZMIJ_USE_U32_CHAIN 1
#define ZMIJ_U32_CHAIN_DIGITS 3
#define ZMIJ_NS zmij_x64_chain32t
#define ZMIJ_SUFFIX zmij_x64_chain32t
#include "zmij_register.inc"
