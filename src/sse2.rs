//! SSE-2 optimized routines
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::mem;

/// SSE2 optimized shuffle for 2-byte type sizes
#[allow(clippy::needless_range_loop)]   // I don't like this suggestion
unsafe fn shuffle2(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8)
{
    const TS: usize = 2;
    const SO128I: usize = mem::size_of::<__m128i>();
    let mut xmm0: [__m128i; 2] = mem::zeroed();
    let mut xmm1: [__m128i; 2] = mem::zeroed();

    for j in (0..vectorizable_elements).step_by(SO128I) {
        /* Fetch 16 elements (32 bytes) then transpose bytes, words and double words. */
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
        /* Transpose quad words */
        xmm1[0] = _mm_unpacklo_epi64(xmm0[0], xmm0[1]);
        xmm1[1] = _mm_unpackhi_epi64(xmm0[0], xmm0[1]);
        /* Store the result vectors */
        let dst_for_jth_element = dst.add(j);
        for k in 0..2 {
            let p = dst_for_jth_element.add(k * total_elements) as *mut __m128i;
            _mm_storeu_si128(p, xmm1[k]);
        }
    }
}

/// SSE2 optimized shuffle for 16-byte type sizes
#[allow(clippy::needless_range_loop)]   // I don't like this suggestion
#[inline(never)]
unsafe fn shuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8)
{
    const TS: usize = 16;
    const SO128I: usize = mem::size_of::<__m128i>();
    let mut xmm0: [__m128i; 16] = mem::zeroed();
    let mut xmm1: [__m128i; 16] = mem::zeroed();

    for j in (0..vectorizable_elements).step_by(SO128I) {
        for k in 0..16 {
            let p = src.add(j * TS + k * SO128I) as *const __m128i;
            xmm0[k] = _mm_loadu_si128(p);
        }
        /* Transpose bytes */
        for k in 0..8 {
            let l = k * 2;
            xmm1[k * 2] = _mm_unpacklo_epi8(xmm0[l], xmm0[l + 1]);
            xmm1[k * 2 + 1] = _mm_unpackhi_epi8(xmm0[l], xmm0[l + 1]);
        }
        /* Transpose words */
        let mut l = 0;
        for k in 0..8 {
            xmm0[k * 2] = _mm_unpacklo_epi16(xmm1[l], xmm1[l + 2]);
            xmm0[k * 2 + 1] = _mm_unpackhi_epi16(xmm1[l], xmm1[l + 2]);
            l += 1;
            if k % 2 == 1 {
                l += 2;
            }
        }
        /* Transpose double words */
        l = 0;
        for k in 0..8 {
            xmm1[k * 2] = _mm_unpacklo_epi32(xmm0[l], xmm0[l + 4]);
            xmm1[k * 2 + 1] = _mm_unpackhi_epi32(xmm0[l], xmm0[l + 4]);
            l += 1;
            if k % 4 == 3 {
                l += 4;
            }
        }
        /* Transpose quad words */
        for k in 0..8 {
            xmm0[k * 2] = _mm_unpacklo_epi64(xmm1[k], xmm1[k + 8]);
            xmm0[k * 2 + 1] = _mm_unpackhi_epi64(xmm1[k], xmm1[k + 8]);
        }
        /* Store the result vectors */
        let dst_for_jth_element = dst.add(j);
        for k in 0..16 {
            let p = dst_for_jth_element.add(k * total_elements) as *mut __m128i;
            _mm_storeu_si128(p, xmm0[k]);
        }
    }
}

pub unsafe fn shuffle(
    typesize: usize,
    len: usize,
    src: *const u8,
    dst: *mut u8)
{
    let vectorized_chunk_size = typesize * mem::size_of::<__m128i>();
    /* If the blocksize is not a multiple of both the typesize and
       the vector size, round the blocksize down to the next value
       which is a multiple of both. The vectorized shuffle can be
       used for that portion of the data, and the naive implementation
       can be used for the remaining portion. */
    let vectorizable_bytes = len - (len % vectorized_chunk_size);
    let vectorizable_elements = vectorizable_bytes / typesize;
    let total_elements = len / typesize;

    /* If the block size is too small to be vectorized,
       use the generic implementation. */
    if len < vectorized_chunk_size {
      crate::generic::shuffle(typesize, len, src, dst);
      return;
    }

    if typesize == 2 {
        shuffle2(vectorizable_elements, total_elements, src, dst);
    } else if typesize == 16 {
        shuffle16(vectorizable_elements, total_elements, src, dst);
    } else {
        //TODO: maybe eliminate optimization for typesize=2, since bfffs does
        //not use it.
        crate::generic::shuffle(typesize, len, src, dst)
    }

    /* If the buffer had any bytes at the end which couldn't be handled
       by the vectorized implementations, use the non-optimized version
       to finish them up. */
    if vectorizable_bytes < len {
        crate::generic::shuffle_partial(typesize, vectorizable_bytes, len, src, dst);
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
        }
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
        fn compare(#[case] typesize: usize, #[case] len: usize) {
            require_sse2!();
            let mut rng = rand::thread_rng();

            let src = (0..len).map(|_| rng.gen()).collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::sse2::shuffle(typesize, len, src.as_ptr(), sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }
    }
}
