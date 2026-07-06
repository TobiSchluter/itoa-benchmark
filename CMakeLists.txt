cmake_minimum_required(VERSION 3.20)
project(itoa_benchmark CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ITOA_NATIVE "Tune for the local CPU (-march=native)" ON)
set(ITOA_ARCH "x86-64-v2" CACHE STRING
    "Baseline -march when ITOA_NATIVE is OFF (needs >=v2 for the SSE4.1 path)")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# gnu++ (not strict c++) so that std::numeric_limits and arithmetic on the
# __int128 extension behave as expected across gcc and clang.
set(CMAKE_CXX_EXTENSIONS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# fmt (fetched, pinned). Provides format_int and __int128 formatting.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        12.2.0
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(fmt)

# ---------------------------------------------------------------------------
# machine.h (CPU model string used in the result filename)
# ---------------------------------------------------------------------------
set(MACHINE_H "${CMAKE_SOURCE_DIR}/src/machine.h")
if(NOT EXISTS "${MACHINE_H}")
    set(_model "unknown")
    if(EXISTS "/proc/cpuinfo")
        file(STRINGS "/proc/cpuinfo" _cpulines REGEX "model name")
        if(_cpulines)
            list(GET _cpulines 0 _line)
            string(REGEX REPLACE "model name.*: " "" _model "${_line}")
            string(REGEX REPLACE "[ \t]+" "" _model "${_model}")
            string(REPLACE "Intel(R)" "" _model "${_model}")
            string(REPLACE "(TM)" "" _model "${_model}")
            string(REPLACE "(R)" "" _model "${_model}")
            string(REPLACE "CPU" "" _model "${_model}")
        endif()
    endif()
    file(WRITE "${MACHINE_H}" "#define MACHINE \"${_model}\"\n")
endif()

# ---------------------------------------------------------------------------
# Benchmark target
# ---------------------------------------------------------------------------
file(GLOB ITOA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
# The zmij variant TUs are compiled separately, each pinned to its own
# microarchitecture level (below), so keep them out of the -march=native target.
list(FILTER ITOA_SOURCES EXCLUDE REGEX "/(zmij_(scalar|sse2|sse41|avx2))\\.cpp$")

add_executable(itoa ${ITOA_SOURCES})
target_include_directories(itoa PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_compile_definitions(itoa PRIVATE __STDC_FORMAT_MACROS=1 NDEBUG)
target_link_libraries(itoa PRIVATE fmt::fmt)

target_compile_options(itoa PRIVATE -O3)
if(ITOA_NATIVE)
    target_compile_options(itoa PRIVATE -march=native)
else()
    target_compile_options(itoa PRIVATE -march=${ITOA_ARCH})
endif()

# ---------------------------------------------------------------------------
# zmij microarchitecture variants.
#
# The library (src/zmij.cc) selects code paths from the compiler's ISA macros,
# so each tier is a separate object library pinned to an x86-64 microarch level
# and linked in. OBJECT libraries add their objects directly to `itoa`, so the
# self-registering (REGISTER_TEST128) TUs are never dropped by the linker.
#   scalar  -march=x86-64 + ZMIJ_USE_SIMD=0   (no SIMD)
#   sse2    -march=x86-64                      (v1, strict SSE2)
#   sse41   -march=x86-64-v2                   (SSE4.2/4.1 + POPCNT, no AVX)
#   avx2    -march=x86-64-v3                   (AVX2 -> 256-bit u128 path)
# ---------------------------------------------------------------------------
function(add_zmij_variant name arch)
    add_library(${name} OBJECT "${CMAKE_SOURCE_DIR}/src/${name}.cpp")
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(${name} PRIVATE __STDC_FORMAT_MACROS=1 NDEBUG)
    target_compile_options(${name} PRIVATE -O3 ${arch})
    target_link_libraries(itoa PRIVATE ${name})
endfunction()

add_zmij_variant(zmij_scalar -march=x86-64)
add_zmij_variant(zmij_sse2   -march=x86-64)
add_zmij_variant(zmij_sse41  -march=x86-64-v2)
add_zmij_variant(zmij_avx2   -march=x86-64-v3)

# The benchmark reads/writes ./result relative to the working dir; run from build.
set_target_properties(itoa PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
