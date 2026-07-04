// {fmt} library implementations.
//
// For 32/64-bit we use fmt::format_int, the dedicated fast integer formatter.
// It formats into an internal buffer; we copy the null-terminated result out.
//
// __int128 has no standard formatting, so for 128-bit we use the general
// fmt::format_to(buffer, "{}", value), which writes straight into our buffer
// and returns the past-the-end pointer.

#include <cstring>
#include <fmt/format.h>
#include "test.h"

template <typename T>
static inline void format_int_to(T value, char* buffer) {
    fmt::format_int f(value);
    // c_str() is null-terminated; copy size()+1 bytes to include the '\0'.
    memcpy(buffer, f.c_str(), f.size() + 1);
}

void u32toa_fmt(uint32_t value, char* buffer) { format_int_to(value, buffer); }
void i32toa_fmt(int32_t  value, char* buffer) { format_int_to(value, buffer); }
void u64toa_fmt(uint64_t value, char* buffer) { format_int_to(value, buffer); }
void i64toa_fmt(int64_t  value, char* buffer) { format_int_to(value, buffer); }

void u128toa_fmt(uint128_t value, char* buffer) {
    *fmt::format_to(buffer, "{}", value) = '\0';
}

void i128toa_fmt(int128_t value, char* buffer) {
    *fmt::format_to(buffer, "{}", value) = '\0';
}

REGISTER_TEST128(fmt);
