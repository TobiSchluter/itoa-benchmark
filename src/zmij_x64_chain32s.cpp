// zmij, same tier as zmij_x64_native, with the branch-free u32 multiply chain
// taking the leading digit first, then three digits per step.
#define ZMIJ_USE_U32_CHAIN 1
#define ZMIJ_U32_CHAIN_DIGITS 3
#define ZMIJ_U32_CHAIN_DIGIT_FIRST 1
#define ZMIJ_NS zmij_x64_chain32s
#define ZMIJ_SUFFIX zmij_x64_chain32s
#include "zmij_register.inc"
