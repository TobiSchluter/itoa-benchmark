// arm_neon.h stand-in for the neon2sse benchmark variant.
//
// zmij.cc includes <arm_neon.h> whenever ZMIJ_USE_NEON is set. Putting this
// directory on the include path ahead of the toolchain's own headers makes that
// include resolve here, so the NEON code path can be compiled -- and
// benchmarked -- on x86 with every intrinsic lowered to SSE by Intel's
// ARM_NEON_2_x86_SSE header (fetched at configure time; see ITOA_NEON2SSE in
// CMakeLists.txt).
//
// This measures the NEON *algorithm* as translated to SSE, not native ARM
// timings: the lowering is not always one instruction per intrinsic, so the
// numbers say how the NEON structure fares on this core, nothing about an
// actual ARM part.
//
// Note on aligned accesses: upstream defines LOAD_SI128/STORE_SI128 as
// _mm_loadu_si128/_mm_storeu_si128, matching the unconditional vld1q/vst1q it
// stands in for. Older revisions selected between the aligned and unaligned
// form with a runtime address test, which put a branch inside every load and
// store and skewed these benchmarks; if the pinned revision is ever moved
// backwards, override both macros to the unaligned form before the include
// below.

#ifndef ZMIJ_NEON2SSE_ARM_NEON_H
#define ZMIJ_NEON2SSE_ARM_NEON_H

#include "NEON_2_SSE.h"

// neon2sse covers ARMv7 NEON, so the AArch64-only intrinsics zmij uses are
// missing. Both map to one SSE instruction, keeping the instruction count the
// same as on ARM.

// vcgtzq_s8: lanewise a > 0.
static inline uint8x16_t vcgtzq_s8(int8x16_t a) {
  return vcgtq_s8(a, vdupq_n_s8(0));
}

// vqtbl1q_u8: byte table lookup, zero where the index is out of range.
//
// pshufb zeroes a lane when the index's high bit is set but otherwise wraps it
// (index & 15), where ARM zeroes anything >= 16. Every index vector zmij feeds
// this -- revalign_shuffle, shift_shuffle, the exp_float_shuffles -- holds only
// 0..15 or 0x80, so the two agree here and no index fixup is needed. Add one
// (or the mask below) before using this for anything else:
//   idx = _mm_or_si128(idx, _mm_cmpgt_epi8(idx, _mm_set1_epi8(15)))
static inline uint8x16_t vqtbl1q_u8(uint8x16_t t, uint8x16_t idx) {
  return _mm_shuffle_epi8(t, idx);
}

#endif  // ZMIJ_NEON2SSE_ARM_NEON_H
