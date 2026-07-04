#include <stdint.h>
#include "test.h"

void u32toa_null(uint32_t, char*) {
}

void i32toa_null(int32_t, char*) {
}

void u64toa_null(uint64_t, char*) {
}

void i64toa_null(int64_t, char*) {
}

void u128toa_null(uint128_t, char*) {
}

void i128toa_null(int128_t, char*) {
}

REGISTER_TEST128(null);
