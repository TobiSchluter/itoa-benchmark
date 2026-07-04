#ifndef ITOA_BENCHMARK_TEST_H
#define ITOA_BENCHMARK_TEST_H

#include <vector>
#include <string.h>
#include <cstdint>

// 128-bit integer support. __int128 is a gcc/clang extension; there is no
// standard formatting for it, so implementations that support it use fmt.
typedef unsigned __int128 uint128_t;
typedef          __int128 int128_t;

struct Test;
typedef std::vector<const Test *> TestList;
class TestManager {
public:
    static TestManager& Instance() {
        static TestManager singleton;
        return singleton;
    }

    void AddTest(const Test* test) {
        mTests.push_back(test);
    }

    const TestList& GetTests() const {
        return mTests;
    }

    TestList& GetTests() {
        return mTests;
    }

private:
    TestList mTests;
};

struct Test {
    Test(
        const char* fname,
        void (*u32toa)(uint32_t, char*),
        void (*i32toa)(int32_t, char*),
        void (*u64toa)(uint64_t, char*),
        void (*i64toa)(int64_t, char*),
        void (*u128toa)(uint128_t, char*) = 0,
        void (*i128toa)(int128_t, char*) = 0)
        :
        fname(fname),
        u32toa(u32toa),
        i32toa(i32toa),
        u64toa(u64toa),
        i64toa(i64toa),
        u128toa(u128toa),
        i128toa(i128toa)
    {
        TestManager::Instance().AddTest(this);
    }

    bool operator<(const Test& rhs) const {
        return strcmp(fname, rhs.fname) < 0;
    }

    const char* fname;
    void (*u32toa)(uint32_t, char*);
    void (*i32toa)(int32_t, char*);
    void (*u64toa)(uint64_t, char*);
    void (*i64toa)(int64_t, char*);
    void (*u128toa)(uint128_t, char*);   // may be null (most impls: no 128-bit)
    void (*i128toa)(int128_t, char*);    // may be null
};


#define STRINGIFY(x) #x

// The extra indirection lets `f` itself be a macro: it is fully expanded before
// the ## token-pasting in the _IMPL forms.

// Register a 32/64-bit-only implementation (128-bit slots left null).
#define REGISTER_TEST(f) REGISTER_TEST_IMPL(f)
#define REGISTER_TEST_IMPL(f) \
    static Test gRegister##f(STRINGIFY(f), \
        u32toa##_##f, i32toa##_##f, u64toa##_##f, i64toa##_##f)

// Register an implementation that also provides 128-bit conversion.
#define REGISTER_TEST128(f) REGISTER_TEST128_IMPL(f)
#define REGISTER_TEST128_IMPL(f) \
    static Test gRegister##f(STRINGIFY(f), \
        u32toa##_##f, i32toa##_##f, u64toa##_##f, i64toa##_##f, \
        u128toa##_##f, i128toa##_##f)

#endif  // ITOA_BENCHMARK_TEST_H
