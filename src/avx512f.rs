//! AVX2 optimized routines
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as simd;
#[cfg(target_arch = "x86")]
use core::arch::x86 as simd;
use simd::{
    __m128i, _mm_i32gather_epi32, _mm_set_epi32, _mm_set_epi8,
    _mm_shuffle_epi8, _mm_i32scatter_epi32
};

use std::mem;

/// AVX512F optimized shuffle for 16-byte type sizes,
#[allow(clippy::needless_range_loop)]   // I don't like this suggestion
#[target_feature(enable = "avx512f")]
unsafe fn shuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8)
{
    assert_eq!(vectorizable_elements % 4, 0);
    assert_eq!(total_elements, vectorizable_elements, "TODO");

    const TS: usize = 16;
    const SO128I: usize = mem::size_of::<__m128i>();
    let mut xmm: __m128i = mem::zeroed();
    let mut xmm1: __m128i = mem::zeroed();

    let loadindex = _mm_set_epi32(3 * TS as i32, 2 * TS as i32, TS as i32, 0);
    let shuf8 = _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
    let storeindex = _mm_set_epi32(
        vectorizable_elements * 3 * TS / 4,
        vectorizable_elements * TS / 2,
        vectorizable_elements * TS / 4,
        0
    );

    for i in 0..vectorizable_elements / 4 {
        for j in 0..4 {
            let p = src.add(i * 4 * TS + j * 4) as *const i32;
            xmm = _mm_i32gather_epi32(p, loadindex, 1);
            // xmm should look like [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
            xmm = _mm_shuffle_epi8(xmm, shuf8);
            // xmm should look like [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51]
            let p = dst.add(i * 4 + j * vectorizable_elements * TS / 4);
            _mm_i32scatter_epi32(p, storeindex, 1);
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

    if typesize == 16 {
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
        todo!()
      // crate::generic::shuffle(typesize, vectorizable_bytes, len, src, dst);
    }
}

#[cfg(test)]
mod t {
    macro_rules! require_avx512f {
        () => {
            if !is_x86_feature_detected!("avx512f") {
                eprintln!("Skipping: AVX512F unavailable.");
                return;
            }
        }
    }

    mod shuffle {
        use rand::Rng;
        use rstest::rstest;
        use super::*;

        #[rstest]
        #[case(16, 256)]
        #[case(16, 4096)]
        #[case(16, 4352)]
        fn compare(#[case] typesize: usize, #[case] len: usize) {
            require_avx512f!();
            let mut rng = rand::thread_rng();

            let src = (0..len).map(|_| rng.gen()).collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::avx512f::shuffle(typesize, len, src.as_ptr(), sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }

    }
}

