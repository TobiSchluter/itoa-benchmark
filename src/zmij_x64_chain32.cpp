// zmij, same tier as zmij_x64_native, with the branch-free u32 multiply chain
// (count-selected left-alignment scale, two digits per step, pair table in
// data).
#define ZMIJ_USE_U32_CHAIN 1
#define ZMIJ_NS zmij_x64_chain32
#define ZMIJ_SUFFIX zmij_x64_chain32
#include "zmij_register.inc"
