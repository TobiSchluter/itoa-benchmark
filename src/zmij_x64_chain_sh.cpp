// zmij, same tier as zmij_x64_native, with the u64 GPR multiply chains and
// the pshufb left-align.
#define ZMIJ_USE_U64_CHAIN 1
#define ZMIJ_U64_CHAIN_SHUFFLE 1
#define ZMIJ_NS zmij_x64_chain_sh
#define ZMIJ_SUFFIX zmij_x64_chain_sh
#include "zmij_register.inc"
