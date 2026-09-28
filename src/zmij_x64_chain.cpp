// zmij, same tier as zmij_x64_native, with the u64 GPR multiply chains and
// the scaled (left-aligned) block stores.
#define ZMIJ_USE_U64_CHAIN 1
#define ZMIJ_NS zmij_x64_chain
#define ZMIJ_SUFFIX zmij_x64_chain
#include "zmij_register.inc"
