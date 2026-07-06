// Toy reference for the byte-printing regime (the `uniform` 0..255 mode): the
// whole conversion is one lookup in a table of 256 pre-rendered strings and a
// fixed 4-byte store. Out-of-range values are deliberately not handled (the
// index is truncated to the low 8 bits), so this entry is skipped by the
// general verification and only its uniform-mode numbers are meaningful.
#include <stdint.h>
#include <string.h>
#include "test.h"

namespace {

struct Table256 {
    // 4 bytes fit "255\0"; zero-padded so a fixed 4-byte copy null-terminates.
    alignas(64) char str[256][4] = {};
    constexpr Table256() {
        for (int i = 0; i < 256; i++) {
            char* p = str[i];
            if (i >= 100) *p++ = char('0' + i / 100);
            if (i >= 10)  *p++ = char('0' + i / 10 % 10);
            *p = char('0' + i % 10);
        }
    }
};

constexpr Table256 kTable;

inline void Lookup(size_t v, char* out) {
    memcpy(out, kTable.str[v & 255], 4);   // one 32-bit store, NUL included
}

}  // namespace

void u32toa_toy256(uint32_t v, char* out)   { Lookup(v, out); }
void i32toa_toy256(int32_t v, char* out)    { Lookup((size_t)(uint32_t)v, out); }
void u64toa_toy256(uint64_t v, char* out)   { Lookup(v, out); }
void i64toa_toy256(int64_t v, char* out)    { Lookup((size_t)(uint64_t)v, out); }
void u128toa_toy256(uint128_t v, char* out) { Lookup((size_t)v, out); }
void i128toa_toy256(int128_t v, char* out)  { Lookup((size_t)(uint128_t)v, out); }

REGISTER_TEST128(toy256);
