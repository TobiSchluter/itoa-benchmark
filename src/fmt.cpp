// {fmt} library implementations.
//
// For 32/64-bit we call fmt::detail::format_decimal directly, the same
// digit-writing primitive fmt::format_int wraps. It writes straight into our
// buffer with no intermediate copy, but requires the digit count up front
// (count_digits), since it fills back-to-front within [buffer, buffer+n).
//
// __int128 has no standard formatting, so for 128-bit we go through the general
// fmt::format_to(buffer, ..., value), which writes straight into our buffer and
// returns the past-the-end pointer. The format string is FMT_COMPILE("{}") so
// that it is parsed at compile time: the lone replacement field lowers to a
// direct, inlinable call to fmt's integer writer, with no runtime format parse
// and no type-erased argument visit. A plain "{}" literal instead calls the
// out-of-line vformat_to, which costs 26-77% (gcc) / 3-32% (clang) more per
// conversion. FMT_COMPILE is gated on __cpp_if_constexpr and degrades silently
// to a runtime format string before C++17, hence the standard set in
// CMakeLists.txt.

#include <fmt/compile.h>
#include <fmt/format.h>
#include "test.h"

// Format an unsigned value straight into buffer; returns the past-the-end ptr.
template <typename UInt>
static inline char* format_unsigned_to(UInt value, char* buffer) {
    int num_digits = fmt::detail::count_digits(value);
    fmt::detail::format_decimal(buffer, value, num_digits);
    return buffer + num_digits;
}

// Signed: emit the sign, then format the magnitude (same idiom as format_int).
template <typename Int>
static inline char* format_signed_to(Int value, char* buffer) {
    auto abs_value = static_cast<fmt::detail::uint32_or_64_or_128_t<Int>>(value);
    *buffer = '-';
    if (value < 0) {
        buffer++;
        abs_value = 0 - abs_value;
    }
    return format_unsigned_to(abs_value, buffer);
}

char* u32toa_r_fmt(uint32_t value, char* buffer) { return format_unsigned_to(value, buffer); }
char* i32toa_r_fmt(int32_t  value, char* buffer) { return format_signed_to(value, buffer); }
char* u64toa_r_fmt(uint64_t value, char* buffer) { return format_unsigned_to(value, buffer); }
char* i64toa_r_fmt(int64_t  value, char* buffer) { return format_signed_to(value, buffer); }
char* u128toa_r_fmt(uint128_t value, char* buffer) { return fmt::format_to(buffer, FMT_COMPILE("{}"), value); }
char* i128toa_r_fmt(int128_t value, char* buffer)  { return fmt::format_to(buffer, FMT_COMPILE("{}"), value); }

// Null-terminating variants: format, then cap with '\0'.
void u32toa_fmt(uint32_t value, char* buffer) { *u32toa_r_fmt(value, buffer) = '\0'; }
void i32toa_fmt(int32_t  value, char* buffer) { *i32toa_r_fmt(value, buffer) = '\0'; }
void u64toa_fmt(uint64_t value, char* buffer) { *u64toa_r_fmt(value, buffer) = '\0'; }
void i64toa_fmt(int64_t  value, char* buffer) { *i64toa_r_fmt(value, buffer) = '\0'; }

void u128toa_fmt(uint128_t value, char* buffer) {
    *fmt::format_to(buffer, FMT_COMPILE("{}"), value) = '\0';
}

void i128toa_fmt(int128_t value, char* buffer) {
    *fmt::format_to(buffer, FMT_COMPILE("{}"), value) = '\0';
}

REGISTER_TEST128(fmt);
