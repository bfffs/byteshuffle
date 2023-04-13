//! AVX2 optimized routines
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as simd;
#[cfg(target_arch = "x86")]
use core::arch::x86 as simd;
use simd::{
    __m128i, _mm_blend_epi32, _mm_i32gather_epi32, _mm_set_epi32, _mm_set_epi8, _mm_shuffle_epi32,
    _mm_shuffle_epi8, _mm_storeu_si128,
};

use std::mem;

/// AVX2 optimized shuffle for 16-byte type sizes, 
#[allow(clippy::needless_range_loop)]   // I don't like this suggestion
#[target_feature(enable = "avx2")]
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
    let mut xmm: [__m128i; 4] = mem::zeroed();
    let mut xmm1: [__m128i; 4] = mem::zeroed();

    let vindex = _mm_set_epi32(3 * SO128I as i32, 2 * SO128I as i32, SO128I as i32, 0);
    let shuf8 = _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);

    for i in 0..vectorizable_elements / 16 {
        for j in (0..16).step_by(4) {
            // Outline:
            // Starting with data that looks like [0, 1, 2, 3, ... 255]
            // for k in 0, 4, 8, 12
            //   * gather-load into 4 __m128i registers that each look like 0123012301230123
            //   * shuffle each so it looks like 0000111122223333
            //   * Use i32 shuffles and blends to get four registers that look like
            //     - 0000000000000000
            //     - 1111111111111111
            //     - 2222222222222222
            //     - 3333333333333333
            //   * Write to dst
            // Tip:
            //   View an __m128i variable in rust-gdb with "p/x *(&xmm1[k] as *const u8)@16"
            for k in 0..4 {
                let p = src.add(i * 16 * TS + j + 4 * TS * k) as *const i32;
                xmm[k] = _mm_i32gather_epi32(p, vindex, 1);
                // xmm[0] should look like [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
                xmm[k] = _mm_shuffle_epi8(xmm[k], shuf8);
                // xmm[0] should look like [0, 16, 32, 48, 1, 17, 33, 49, 3, 18, 34, 50, 3, 19, 35, 51]
            }
            // This next step is logically a rotation, but we use shuffle instructions
            xmm[1] = _mm_shuffle_epi32(xmm[1], 0x93);
            xmm[2] = _mm_shuffle_epi32(xmm[2], 0x4e);
            xmm[3] = _mm_shuffle_epi32(xmm[3], 0x39);
            // xmm[0] should be unchanged, xmm[1] rotated 4 bytes right, xmm[2] 8 bytes right, etc

            xmm1[0] = _mm_blend_epi32(xmm[0], xmm[1], 0b0010);
            xmm1[0] = _mm_blend_epi32(xmm1[0], xmm[2], 0b0100);
            xmm1[0] = _mm_blend_epi32(xmm1[0], xmm[3], 0b1000);
            xmm1[1] = _mm_blend_epi32(xmm[0], xmm[1], 0b0100);
            xmm1[1] = _mm_blend_epi32(xmm1[1], xmm[2], 0b1000);
            xmm1[1] = _mm_blend_epi32(xmm1[1], xmm[3], 0b0001);
            xmm1[2] = _mm_blend_epi32(xmm[0], xmm[1], 0b1000);
            xmm1[2] = _mm_blend_epi32(xmm1[2], xmm[2], 0b0001);
            xmm1[2] = _mm_blend_epi32(xmm1[2], xmm[3], 0b0010);
            xmm1[3] = _mm_blend_epi32(xmm[0], xmm[1], 0b0001);
            xmm1[3] = _mm_blend_epi32(xmm1[3], xmm[2], 0b0010);
            xmm1[3] = _mm_blend_epi32(xmm1[3], xmm[3], 0b0100);

            // Now to rotate the xmm1 registers, again using shuffle instructions
            xmm1[1] = _mm_shuffle_epi32(xmm1[1], 0x39);
            xmm1[2] = _mm_shuffle_epi32(xmm1[2], 0x4e);
            xmm1[3] = _mm_shuffle_epi32(xmm1[3], 0x93);

            // Now write the results
            for k in 0..4 {
                let p = dst.add((i + (j + k) * vectorizable_elements / 16) * TS) as *mut __m128i;
                _mm_storeu_si128(p, xmm1[k]);
            }
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
    mod shuffle {
        // use pretty_assertions::assert_eq;
        use rand::Rng;
        use rstest::rstest;

        #[rstest]
        #[case(16, 256)]
        #[case(16, 4096)]
        #[case(16, 4352)]
        fn compare(#[case] typesize: usize, #[case] len: usize) {
            let mut rng = rand::thread_rng();

            let src = (0..len).map(|_| rng.gen()).collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::avx2::shuffle(typesize, len, src.as_ptr(), sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }

        // This test is redundant with the randomly generated one, but easier to debug.
        #[rstest]
        fn compare16x256() {
            let typesize = 16;
            let len = 256;
            let src = (0..=255).collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::avx2::shuffle(typesize, len, src.as_ptr(), sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }

        #[rstest]
        fn compare16x512() {
            let typesize = 16;
            let len = 512;
            let src = (0..len)
                .map(|i| (i % 256) as u8)
                .collect::<Vec<u8>>();
            // let src = (0u16..256)
            //     .collect::<Vec<u16>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            let srcp = src.as_ptr() as *const u8;
            unsafe {
                crate::generic::shuffle(typesize, len, srcp, generic_dst.as_mut_ptr());
                // crate::avx2::shuffle(typesize, 256, src.as_ptr(), sse2_dst.as_mut_ptr());
                crate::avx2::shuffle(typesize, len, srcp, sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }
    }
}
