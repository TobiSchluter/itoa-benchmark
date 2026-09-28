// zmij, same tier as zmij_x64_native, with the u64 GPR multiply chains, the
// digits staged in a stack buffer and left-aligned by one pshufb.
#define ZMIJ_USE_U64_CHAIN 1
#define ZMIJ_U64_CHAIN_SHUFFLE 2
#define ZMIJ_NS zmij_x64_chain_st
#define ZMIJ_SUFFIX zmij_x64_chain_st
#include "zmij_register.inc"
