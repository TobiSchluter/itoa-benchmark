cmake_minimum_required(VERSION 3.20)
project(itoa_benchmark CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ITOA_NATIVE "Tune for the local CPU (-march=native)" ON)
set(ITOA_ARCH "x86-64-v2" CACHE STRING
    "Baseline -march when ITOA_NATIVE is OFF (needs >=v2 for the SSE4.1 path)")
option(ITOA_NEON2SSE
    "On x86, also build zmij's NEON path via the neon2sse header (needs network \
at configure time to fetch it)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# gnu++ (not strict c++) so that std::numeric_limits and arithmetic on the
# __int128 extension behave as expected across gcc and clang.
set(CMAKE_CXX_EXTENSIONS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# fmt (fetched, pinned). Provides format_int and __int128 formatting.
# Pinned past 12.2.0 for the 128-bit formatting optimization (fmtlib/fmt 7b4ef1c8).
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        7b4ef1c8145f21e126167f52a594909dde8c89fe)
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
list(FILTER ITOA_SOURCES EXCLUDE REGEX "/(zmij_(scalar|x64_v1|x64_v2|x64_v3|x64_v4|x64_native|neon|neon_shim))\\.cpp$")

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
# The library (src/zmij-int.cc) selects code paths from the compiler's ISA macros,
# so each tier is a separate object library pinned to an x86-64 microarch level
# and linked in. OBJECT libraries add their objects directly to `itoa`, so the
# self-registering (REGISTER_TEST128) TUs are never dropped by the linker.
#   zmij_scalar          -march=x86-64 + ZMIJ_USE_SIMD=0  (no SIMD)
#   zmij_x64_v1          -march=x86-64                     (strict SSE2)
#   zmij_x64_v2          -march=x86-64-v2                  (SSE4.2/4.1 + POPCNT)
#   zmij_x64_v3          -march=x86-64-v3                  (AVX2 -> 256-bit u128 kernel)
#   zmij_x64_v4          -march=x86-64-v4                  (AVX-512F/BW/DQ/VL codegen)
#   zmij_x64_native      -march=native -mtune=native       (tuned for build host)
# ---------------------------------------------------------------------------
function(add_zmij_variant name arch)
    add_library(${name} OBJECT "${CMAKE_SOURCE_DIR}/src/${name}.cpp")
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(${name} PRIVATE __STDC_FORMAT_MACROS=1 NDEBUG)
    # Fixed function alignment so A/B comparisons between variants are not
    # skewed by fetch-group luck (arm64 clang only 4-byte-aligns functions;
    # byte-identical entry points have measured ~1 cycle/call apart from
    # start-offset alone).
    target_compile_options(${name} PRIVATE -O3 -falign-functions=64 ${arch})
    target_link_libraries(itoa PRIVATE ${name})
endfunction()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    if(ITOA_NATIVE)
        add_zmij_variant(zmij_scalar  -march=native)
        add_zmij_variant(zmij_neon    -march=native)
    else()
        add_zmij_variant(zmij_scalar  "")
        add_zmij_variant(zmij_neon    "")
    endif()
else()
    add_zmij_variant(zmij_scalar          -march=x86-64)
    add_zmij_variant(zmij_x64_v1          -march=x86-64)
    add_zmij_variant(zmij_x64_v2          -march=x86-64-v2)
    add_zmij_variant(zmij_x64_v3          -march=x86-64-v3)
    add_zmij_variant(zmij_x64_v4          -march=x86-64-v4)
    add_zmij_variant(zmij_x64_native "-march=native;-mtune=native")

    # zmij's NEON path, lowered to SSE by Intel's neon2sse header so it can be
    # benchmarked beside the x64 tiers. Off by default: it needs the header
    # fetched at configure time, and the timings characterize the NEON
    # algorithm on this core rather than any ARM part (see
    # src/neon2sse/arm_neon.h).
    if(ITOA_NEON2SSE)
        set(NEON2SSE_DIR "${CMAKE_BINARY_DIR}/neon2sse")
        set(NEON2SSE_HDR "${NEON2SSE_DIR}/NEON_2_SSE.h")
        if(NOT EXISTS "${NEON2SSE_HDR}")
            message(STATUS "Fetching NEON_2_SSE.h")
            file(DOWNLOAD
                "https://raw.githubusercontent.com/intel/ARM_NEON_2_x86_SSE/472c986aced1e498b84f96f6a9a583f93c4c7bfe/NEON_2_SSE.h"
                "${NEON2SSE_HDR}"
                EXPECTED_HASH SHA256=a13a27dc305c2b984703ca0801fd5d3015c9ea4c6a70394bbf22ce982cbd460e
                TLS_VERIFY ON
                STATUS _n2s_status)
            list(GET _n2s_status 0 _n2s_code)
            if(NOT _n2s_code EQUAL 0)
                list(GET _n2s_status 1 _n2s_msg)
                file(REMOVE "${NEON2SSE_HDR}")
                message(FATAL_ERROR "ITOA_NEON2SSE: download failed: ${_n2s_msg}")
            endif()
        endif()
        add_zmij_variant(zmij_neon_shim "-march=native;-mtune=native")
        target_include_directories(zmij_neon_shim BEFORE PRIVATE
            "${CMAKE_SOURCE_DIR}/src/neon2sse" "${NEON2SSE_DIR}")
        # ZMIJ_USE_SSE=0 so zmij-int.cc takes the NEON branch rather than its own
        # x86 one; the shim still compiles it all down to SSE.
        target_compile_definitions(zmij_neon_shim PRIVATE
            ZMIJ_USE_NEON=1 ZMIJ_USE_SSE=0 ZMIJ_NEON2SSE_SHIM=1)
        # The header trips these on both compilers; they are upstream's
        # concern, not this benchmark's.
        target_compile_options(zmij_neon_shim PRIVATE
            -Wno-deprecated-declarations -Wno-ignored-attributes)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(zmij_neon_shim PRIVATE
                -Wno-gnu-inline-cpp-without-extern)
        endif()
    endif()
endif()

# The benchmark reads/writes ./result relative to the working dir; run from build.
set_target_properties(itoa PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# ---------------------------------------------------------------------------
# Exhaustive int32 and uint32 validation of zmij against jeaiii, timing
# several validation strategies (see src/validate32.cc).
# ---------------------------------------------------------------------------
add_executable(validate32
    "${CMAKE_SOURCE_DIR}/src/validate32.cc"
    "${CMAKE_SOURCE_DIR}/src/itoa_jeaiii.cpp"
    "${CMAKE_SOURCE_DIR}/src/fmt.cpp")
target_include_directories(validate32 PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_compile_definitions(validate32 PRIVATE NDEBUG)
find_package(Threads REQUIRED)
target_link_libraries(validate32 PRIVATE fmt::fmt Threads::Threads)
target_compile_options(validate32 PRIVATE -O3)
if(ITOA_NATIVE)
    target_compile_options(validate32 PRIVATE -march=native)
else()
    target_compile_options(validate32 PRIVATE -march=${ITOA_ARCH})
endif()
