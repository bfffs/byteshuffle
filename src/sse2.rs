//! SSE-2 optimized routines
#[cfg(target_arch = "x86")]
use core::arch::x86 as simd;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as simd;
use std::mem;

use simd::{
    __m128i,
    _mm_loadu_si128,
    _mm_shuffle_epi32,
    _mm_shufflehi_epi16,
    _mm_shufflelo_epi16,
    _mm_storeu_si128,
    _mm_unpackhi_epi16,
    _mm_unpackhi_epi32,
    _mm_unpackhi_epi64,
    _mm_unpackhi_epi8,
    _mm_unpacklo_epi16,
    _mm_unpacklo_epi32,
    _mm_unpacklo_epi64,
    _mm_unpacklo_epi8,
};

const SO128I: usize = mem::size_of::<__m128i>();

/// SSE2 optimized shuffle for 2-byte type sizes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
unsafe fn shuffle2(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 2;
    let mut xmm0: [__m128i; 2] = mem::zeroed();
    let mut xmm1: [__m128i; 2] = mem::zeroed();

    for j in (0..vectorizable_elements).step_by(SO128I) {
        // Fetch 16 elements (32 bytes) then transpose bytes, words and double words.
        for k in 0..2 {
            let p = src.add(j * TS + k * SO128I) as *const __m128i;
            xmm0[k] = _mm_loadu_si128(p);
            xmm0[k] = _mm_shufflelo_epi16(xmm0[k], 0xd8);
            xmm0[k] = _mm_shufflehi_epi16(xmm0[k], 0xd8);
            xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
            xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0x4e);
            xmm0[k] = _mm_unpacklo_epi8(xmm0[k], xmm1[k]);
            xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
            xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0x4e);
            xmm0[k] = _mm_unpacklo_epi16(xmm0[k], xmm1[k]);
            xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
        }
        // Transpose quad words
        xmm1[0] = _mm_unpacklo_epi64(xmm0[0], xmm0[1]);
        xmm1[1] = _mm_unpackhi_epi64(xmm0[0], xmm0[1]);
        // Store the result vectors
        let dst_for_jth_element = dst.add(j);
        for k in 0..2 {
            let p = dst_for_jth_element.add(k * total_elements) as *mut __m128i;
            _mm_storeu_si128(p, xmm1[k]);
        }
    }
}

/// SSE2 optimized shuffle for 16-byte type sizes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
#[inline(never)]
unsafe fn shuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 16;
    let mut xmm0: [__m128i; 16] = mem::zeroed();
    let mut xmm1: [__m128i; 16] = mem::zeroed();

    for j in (0..vectorizable_elements).step_by(SO128I) {
        for k in 0..16 {
            let p = src.add(j * TS + k * SO128I) as *const __m128i;
            xmm0[k] = _mm_loadu_si128(p);
        }
        // Transpose bytes
        for k in 0..8 {
            let l = k * 2;
            xmm1[k * 2] = _mm_unpacklo_epi8(xmm0[l], xmm0[l + 1]);
            xmm1[k * 2 + 1] = _mm_unpackhi_epi8(xmm0[l], xmm0[l + 1]);
        }
        // Transpose words
        let mut l = 0;
        for k in 0..8 {
            xmm0[k * 2] = _mm_unpacklo_epi16(xmm1[l], xmm1[l + 2]);
            xmm0[k * 2 + 1] = _mm_unpackhi_epi16(xmm1[l], xmm1[l + 2]);
            l += 1;
            if k % 2 == 1 {
                l += 2;
            }
        }
        // Transpose double words
        l = 0;
        for k in 0..8 {
            xmm1[k * 2] = _mm_unpacklo_epi32(xmm0[l], xmm0[l + 4]);
            xmm1[k * 2 + 1] = _mm_unpackhi_epi32(xmm0[l], xmm0[l + 4]);
            l += 1;
            if k % 4 == 3 {
                l += 4;
            }
        }
        // Transpose quad words
        for k in 0..8 {
            xmm0[k * 2] = _mm_unpacklo_epi64(xmm1[k], xmm1[k + 8]);
            xmm0[k * 2 + 1] = _mm_unpackhi_epi64(xmm1[k], xmm1[k + 8]);
        }
        // Store the result vectors
        let dst_for_jth_element = dst.add(j);
        for k in 0..16 {
            let p = dst_for_jth_element.add(k * total_elements) as *mut __m128i;
            _mm_storeu_si128(p, xmm0[k]);
        }
    }
}

/// SSE2 optimized shuffle for type sizes larger than 16 bytes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
unsafe fn shuffle_tiled(
    vectorizable_elements: usize,
    total_elements: usize,
    ts: usize,
    src: *const u8,
    dst: *mut u8,
) {
    let mut xmm0: [__m128i; 16] = mem::zeroed();
    let mut xmm1: [__m128i; 16] = mem::zeroed();
    let vecs_rem = ts % SO128I;

    for j in (0..vectorizable_elements).step_by(SO128I) {
        // Advance the offset into the type by the vector size (in bytes), unless this is
        // the initial iteration and the type size is not a multiple of the vector size.
        // In that case, only advance by the number of bytes necessary so that the number
        // of remaining bytes in the type will be a multiple of the vector size.
        let mut off_in_ty = 0;
        while off_in_ty < ts {
            // Fetch elements in groups of 256 bytes
            for k in 0..16 {
                xmm0[k] = _mm_loadu_si128(src.add(off_in_ty + (j + k) * ts) as *const __m128i);
            }
            // Transpose bytes
            for k in 0..8 {
                let l = k * 2;
                xmm1[k * 2] = _mm_unpacklo_epi8(xmm0[l], xmm0[l + 1]);
                xmm1[k * 2 + 1] = _mm_unpackhi_epi8(xmm0[l], xmm0[l + 1]);
            }
            // Transpose words
            let mut l = 0;
            for k in 0..8 {
                xmm0[k * 2] = _mm_unpacklo_epi16(xmm1[l], xmm1[l + 2]);
                xmm0[k * 2 + 1] = _mm_unpackhi_epi16(xmm1[l], xmm1[l + 2]);
                l += 1;
                if (k % 2) == 1 {
                    l += 2;
                }
            }
            // Transpose double words
            let mut l = 0;
            for k in 0..8 {
                xmm1[k * 2] = _mm_unpacklo_epi32(xmm0[l], xmm0[l + 4]);
                xmm1[k * 2 + 1] = _mm_unpackhi_epi32(xmm0[l], xmm0[l + 4]);
                l += 1;
                if (k % 4) == 3 {
                    l += 4;
                }
            }
            // Transpose quad words
            for k in 0..8 {
                xmm0[k * 2] = _mm_unpacklo_epi64(xmm1[k], xmm1[k + 8]);
                xmm0[k * 2 + 1] = _mm_unpackhi_epi64(xmm1[k], xmm1[k + 8]);
            }
            // Store the result vectors
            for k in 0..16 {
                _mm_storeu_si128(
                    dst.add(j + total_elements * (off_in_ty + k)) as *mut __m128i,
                    xmm0[k],
                );
            }
            off_in_ty += if off_in_ty == 0 && vecs_rem > 0 {
                vecs_rem
            } else {
                SO128I
            }
        }
    }
}

pub unsafe fn shuffle(typesize: usize, len: usize, src: *const u8, dst: *mut u8) {
    let vectorized_chunk_size = typesize * mem::size_of::<__m128i>();
    // If the blocksize is not a multiple of both the typesize and
    // the vector size, round the blocksize down to the next value
    // which is a multiple of both. The vectorized shuffle can be
    // used for that portion of the data, and the naive implementation
    // can be used for the remaining portion.
    let vectorizable_bytes = len - (len % vectorized_chunk_size);
    let vectorizable_elements = vectorizable_bytes / typesize;
    let total_elements = len / typesize;

    // If the block size is too small to be vectorized,
    // use the generic implementation.
    if len < vectorized_chunk_size {
        crate::generic::shuffle(typesize, len, src, dst);
        return;
    }

    if typesize == 2 {
        shuffle2(vectorizable_elements, total_elements, src, dst);
    } else if typesize == 16 {
        shuffle16(vectorizable_elements, total_elements, src, dst);
    } else if typesize > SO128I {
        shuffle_tiled(vectorizable_elements, total_elements, typesize, src, dst);
    } else {
        crate::generic::shuffle(typesize, len, src, dst);
        // The generic routine leaves no remainder left to shuffle
        return;
    }

    // If the buffer had any bytes at the end which couldn't be handled
    // by the vectorized implementations, use the non-optimized version
    // to finish them up.
    if vectorizable_bytes < len {
        crate::generic::shuffle_partial(typesize, vectorizable_bytes, len, src, dst);
    }
}

/// SSE2 optimized unshuffle for 2-byte type sizes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
unsafe fn unshuffle2(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 2;
    let mut xmm0: [__m128i; 2] = mem::zeroed();
    let mut xmm1: [__m128i; 2] = mem::zeroed();

    for i in (0..vectorizable_elements).step_by(SO128I) {
        // Load 16 elements (32 bytes) into 2 XMM registers.
        for j in 0..2 {
            let p = src.add(i + j * total_elements) as *const __m128i;
            xmm0[j] = _mm_loadu_si128(p);
        }
        // Shuffle bytes
        // Compute the low 32 bytes
        xmm1[0] = _mm_unpacklo_epi8(xmm0[0], xmm0[1]);
        // Compute the hi 32 bytes
        xmm1[1] = _mm_unpackhi_epi8(xmm0[0], xmm0[1]);
        // Store the result vectors in proper order
        //_mm_storeu_si128(dst.add(i * TS + 0 * SO128I) as *mut __m128i, xmm1[0]);
        //_mm_storeu_si128(dst.add(i * TS + 1 * SO128I) as *mut __m128i, xmm1[1]);
        for k in 0..2 {
            let p = dst.add(i * TS + k * SO128I) as *mut __m128i;
            _mm_storeu_si128(p, xmm1[k]);
        }
    }
}

/// SSE2 optimized unshuffle for 16-byte type sizes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
unsafe fn unshuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 16;
    let mut xmm1: [__m128i; 16] = mem::zeroed();
    let mut xmm2: [__m128i; 16] = mem::zeroed();

    for i in (0..vectorizable_elements).step_by(SO128I) {
        // Load 16 elements (256 bytes) into 16 XMM registers.
        for j in 0..16 {
            let p = src.add(i + j * total_elements) as *const __m128i;
            xmm1[j] = _mm_loadu_si128(p);
        }
        // Shuffle bytes
        for j in 0..8 {
            // Compute the low 32 bytes
            xmm2[j] = _mm_unpacklo_epi8(xmm1[j * 2], xmm1[j * 2 + 1]);
            // Compute the hi 32 bytes
            xmm2[8 + j] = _mm_unpackhi_epi8(xmm1[j * 2], xmm1[j * 2 + 1]);
        }
        // Shuffle 2-byte words
        for j in 0..8 {
            // Compute the low 32 bytes
            xmm1[j] = _mm_unpacklo_epi16(xmm2[j * 2], xmm2[j * 2 + 1]);
            // Compute the hi 32 bytes
            xmm1[8 + j] = _mm_unpackhi_epi16(xmm2[j * 2], xmm2[j * 2 + 1]);
        }
        // Shuffle 4-byte dwords
        for j in 0..8 {
            // Compute the low 32 bytes
            xmm2[j] = _mm_unpacklo_epi32(xmm1[j * 2], xmm1[j * 2 + 1]);
            // Compute the hi 32 bytes
            xmm2[8 + j] = _mm_unpackhi_epi32(xmm1[j * 2], xmm1[j * 2 + 1]);
        }
        // Shuffle 8-byte qwords
        for j in 0..8 {
            // Compute the low 32 bytes
            xmm1[j] = _mm_unpacklo_epi64(xmm2[j * 2], xmm2[j * 2 + 1]);
            // Compute the hi 32 bytes
            xmm1[8 + j] = _mm_unpackhi_epi64(xmm2[j * 2], xmm2[j * 2 + 1]);
        }

        // Store the result vectors in proper order
        #[allow(clippy::erasing_op)]
        _mm_storeu_si128(dst.add(i * TS + 0 * SO128I) as *mut __m128i, xmm1[0]);
        #[allow(clippy::identity_op)]
        _mm_storeu_si128(dst.add(i * TS + 1 * SO128I) as *mut __m128i, xmm1[8]);
        _mm_storeu_si128(dst.add(i * TS + 2 * SO128I) as *mut __m128i, xmm1[4]);
        _mm_storeu_si128(dst.add(i * TS + 3 * SO128I) as *mut __m128i, xmm1[12]);
        _mm_storeu_si128(dst.add(i * TS + 4 * SO128I) as *mut __m128i, xmm1[2]);
        _mm_storeu_si128(dst.add(i * TS + 5 * SO128I) as *mut __m128i, xmm1[10]);
        _mm_storeu_si128(dst.add(i * TS + 6 * SO128I) as *mut __m128i, xmm1[6]);
        _mm_storeu_si128(dst.add(i * TS + 7 * SO128I) as *mut __m128i, xmm1[14]);
        _mm_storeu_si128(dst.add(i * TS + 8 * SO128I) as *mut __m128i, xmm1[1]);
        _mm_storeu_si128(dst.add(i * TS + 9 * SO128I) as *mut __m128i, xmm1[9]);
        _mm_storeu_si128(dst.add(i * TS + 10 * SO128I) as *mut __m128i, xmm1[5]);
        _mm_storeu_si128(dst.add(i * TS + 11 * SO128I) as *mut __m128i, xmm1[13]);
        _mm_storeu_si128(dst.add(i * TS + 12 * SO128I) as *mut __m128i, xmm1[3]);
        _mm_storeu_si128(dst.add(i * TS + 13 * SO128I) as *mut __m128i, xmm1[11]);
        _mm_storeu_si128(dst.add(i * TS + 14 * SO128I) as *mut __m128i, xmm1[7]);
        _mm_storeu_si128(dst.add(i * TS + 15 * SO128I) as *mut __m128i, xmm1[15]);
    }
}

/// SSE2 optimized unshuffle for type sizes larger than 16 bytes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
unsafe fn unshuffle_tiled(
    vectorizable_elements: usize,
    total_elements: usize,
    ts: usize,
    src: *const u8,
    dst: *mut u8,
) {
    let mut xmm1: [__m128i; 16] = mem::zeroed();
    let mut xmm2: [__m128i; 16] = mem::zeroed();
    let vecs_rem = ts % SO128I;

    // The unshuffle loops are inverted (compared to shuffle_tiled16_sse2)
    // to optimize cache utilization.
    let mut off_in_ty = 0;
    while off_in_ty < ts {
        for i in (0..vectorizable_elements).step_by(SO128I) {
            // Load the first 128 bytes in 16 XMM registers
            for j in 0..16 {
                let p = src.add(i + total_elements * (off_in_ty + j)) as *const __m128i;
                xmm1[j] = _mm_loadu_si128(p);
            }
            // Shuffle bytes
            for j in 0..8 {
                // Compute the low 32 bytes
                xmm2[j] = _mm_unpacklo_epi8(xmm1[j * 2], xmm1[j * 2 + 1]);
                // Compute the hi 32 bytes
                xmm2[8 + j] = _mm_unpackhi_epi8(xmm1[j * 2], xmm1[j * 2 + 1]);
            }
            // Shuffle 2-byte words
            for j in 0..8 {
                // Compute the low 32 bytes
                xmm1[j] = _mm_unpacklo_epi16(xmm2[j * 2], xmm2[j * 2 + 1]);
                // Compute the hi 32 bytes
                xmm1[8 + j] = _mm_unpackhi_epi16(xmm2[j * 2], xmm2[j * 2 + 1]);
            }
            // Shuffle 4-byte dwords
            for j in 0..8 {
                // Compute the low 32 bytes
                xmm2[j] = _mm_unpacklo_epi32(xmm1[j * 2], xmm1[j * 2 + 1]);
                // Compute the hi 32 bytes
                xmm2[8 + j] = _mm_unpackhi_epi32(xmm1[j * 2], xmm1[j * 2 + 1]);
            }
            // Shuffle 8-byte qwords
            for j in 0..8 {
                // Compute the low 32 bytes
                xmm1[j] = _mm_unpacklo_epi64(xmm2[j * 2], xmm2[j * 2 + 1]);
                // Compute the hi 32 bytes
                xmm1[8 + j] = _mm_unpackhi_epi64(xmm2[j * 2], xmm2[j * 2 + 1]);
            }
            // Store the result vectors in proper order
            #[allow(clippy::identity_op)]
            _mm_storeu_si128(dst.add(off_in_ty + (i + 0) * ts) as *mut __m128i, xmm1[0]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 1) * ts) as *mut __m128i, xmm1[8]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 2) * ts) as *mut __m128i, xmm1[4]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 3) * ts) as *mut __m128i, xmm1[12]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 4) * ts) as *mut __m128i, xmm1[2]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 5) * ts) as *mut __m128i, xmm1[10]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 6) * ts) as *mut __m128i, xmm1[6]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 7) * ts) as *mut __m128i, xmm1[14]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 8) * ts) as *mut __m128i, xmm1[1]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 9) * ts) as *mut __m128i, xmm1[9]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 10) * ts) as *mut __m128i, xmm1[5]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 11) * ts) as *mut __m128i, xmm1[13]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 12) * ts) as *mut __m128i, xmm1[3]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 13) * ts) as *mut __m128i, xmm1[11]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 14) * ts) as *mut __m128i, xmm1[7]);
            _mm_storeu_si128(dst.add(off_in_ty + (i + 15) * ts) as *mut __m128i, xmm1[15]);
        }
        off_in_ty += if off_in_ty == 0 && vecs_rem > 0 {
            vecs_rem
        } else {
            SO128I
        };
    }
}

pub unsafe fn unshuffle(typesize: usize, len: usize, src: *const u8, dst: *mut u8) {
    let vectorized_chunk_size = typesize * mem::size_of::<__m128i>();
    // If the blocksize is not a multiple of both the typesize and
    // the vector size, round the blocksize down to the next value
    // which is a multiple of both. The vectorized unshuffle can be
    // used for that portion of the data, and the naive implementation
    // can be used for the remaining portion.
    let vectorizable_bytes = len - (len % vectorized_chunk_size);
    let vectorizable_elements = vectorizable_bytes / typesize;
    let total_elements = len / typesize;

    // If the block size is too small to be vectorized,
    // use the generic implementation.
    if len < vectorized_chunk_size {
        crate::generic::unshuffle(typesize, len, src, dst);
        return;
    }

    if typesize == 2 {
        unshuffle2(vectorizable_elements, total_elements, src, dst);
    } else if typesize == 16 {
        unshuffle16(vectorizable_elements, total_elements, src, dst);
    } else if typesize > SO128I {
        unshuffle_tiled(vectorizable_elements, total_elements, typesize, src, dst);
    } else {
        crate::generic::unshuffle(typesize, len, src, dst);
        // The generic routine leaves no remainder left to shuffle
        return;
    }

    // If the buffer had any bytes at the end which couldn't be handled
    // by the vectorized implementations, use the non-optimized version
    // to finish them up.
    if vectorizable_bytes < len {
        crate::generic::unshuffle_partial(typesize, vectorizable_bytes, len, src, dst);
    }
}

#[cfg(test)]
mod t {
    macro_rules! require_sse2 {
        () => {
            if !is_x86_feature_detected!("sse2") {
                eprintln!("Skipping: SSE2 unavailable.");
                return;
            }
        };
    }

    mod shuffle {
        use rand::Rng;
        use rstest::rstest;

        #[rstest]
        #[case(2, 16)]
        #[case(2, 32)]
        #[case(2, 64)]
        #[case(2, 4096)]
        #[case(16, 16)]
        #[case(16, 64)]
        #[case(16, 128)]
        #[case(16, 256)]
        #[case(16, 4096)]
        #[case(18, 288)]
        fn compare(#[case] typesize: usize, #[case] len: usize) {
            require_sse2!();
            let mut rng = rand::rng();

            let src = (0..len).map(|_| rng.random()).collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::sse2::shuffle(typesize, len, src.as_ptr(), sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }
    }

    mod unshuffle {
        use rand::Rng;
        use rstest::rstest;

        #[rstest]
        #[case(2, 16)]
        #[case(2, 32)]
        #[case(2, 64)]
        #[case(2, 4096)]
        #[case(16, 16)]
        #[case(16, 64)]
        #[case(16, 128)]
        #[case(16, 256)]
        #[case(16, 4096)]
        fn compare(#[case] typesize: usize, #[case] len: usize) {
            require_sse2!();
            let mut rng = rand::rng();

            let src = (0..len).map(|_| rng.random()).collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            unsafe {
                crate::generic::unshuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::sse2::unshuffle(typesize, len, src.as_ptr(), sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }
    }
}
