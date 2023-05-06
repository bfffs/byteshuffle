//! AVX2 optimized routines
#[cfg(target_arch = "x86")]
use core::arch::x86 as simd;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as simd;
use std::mem;

use simd::{
    __m128i,
    __m256i,
    _mm256_blend_epi32,
    _mm256_loadu2_m128i,
    _mm256_loadu_si256,
    _mm256_permute4x64_epi64,
    _mm256_set_epi8,
    _mm256_shuffle_epi8,
    _mm256_storeu_si256,
    _mm256_unpackhi_epi16,
    _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64,
    _mm256_unpackhi_epi8,
    _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32,
    _mm256_unpacklo_epi64,
    _mm256_unpacklo_epi8,
};

const SO256I: usize = mem::size_of::<__m256i>();
const SO128I: usize = mem::size_of::<__m128i>();

/// AVX2 optimized shuffle for 2-byte type sizes,
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
#[target_feature(enable = "avx2")]
unsafe fn shuffle2(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 2;
    let mut ymm0: [__m256i; 16] = mem::zeroed();
    let mut ymm1: [__m256i; 16] = mem::zeroed();
    // Create the shuffle mask.
    // NOTE: The XMM/YMM 'set' intrinsics require the arguments to be ordered from
    // most to least significant (i.e., their order is reversed when compared to
    // loading the mask from an array).
    #[rustfmt::skip]
    let shmask = _mm256_set_epi8(
        0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
        0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
        0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
        0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00);

    for j in (0..vectorizable_elements).step_by(SO256I) {
        // Fetch 32 elements (64 bytes) then transpose bytes, words and double words.
        for k in 0..2 {
            let p = src.add(j * TS + k * SO256I) as *const __m256i;
            ymm0[k] = _mm256_loadu_si256(p);
            ymm1[k] = _mm256_shuffle_epi8(ymm0[k], shmask);
        }

        ymm0[0] = _mm256_permute4x64_epi64(ymm1[0], 0xd8);
        ymm0[1] = _mm256_permute4x64_epi64(ymm1[1], 0x8d);

        ymm1[0] = _mm256_blend_epi32(ymm0[0], ymm0[1], 0xf0);
        ymm0[1] = _mm256_blend_epi32(ymm0[0], ymm0[1], 0x0f);
        ymm1[1] = _mm256_permute4x64_epi64(ymm0[1], 0x4e);

        // Store the result vectors
        for k in 0..2 {
            let p = dst.add(j + k * total_elements) as *mut __m256i;
            _mm256_storeu_si256(p, ymm1[k]);
        }
    }
}

/// AVX2 optimized shuffle for 16-byte type sizes,
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
#[target_feature(enable = "avx2")]
unsafe fn shuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 16;
    let mut ymm0: [__m256i; 16] = mem::zeroed();
    let mut ymm1: [__m256i; 16] = mem::zeroed();

    // Create the shuffle mask.
    // NOTE: The XMM/YMM 'set' intrinsics require the arguments to be ordered from
    // most to least significant (i.e., their order is reversed when compared to
    // loading the mask from an array).
    #[rustfmt::skip]
    let shmask: __m256i = _mm256_set_epi8(
        0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04,
        0x0b, 0x03, 0x0a, 0x02, 0x09, 0x01, 0x08, 0x00,
        0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04,
        0x0b, 0x03, 0x0a, 0x02, 0x09, 0x01, 0x08, 0x00);

    for j in (0..vectorizable_elements).step_by(SO256I) {
        // Fetch 32 elements (512 bytes) into 16 YMM registers.
        for k in 0..16 {
            let p = src.add(j * TS + k * SO256I) as *const __m256i;
            ymm0[k] = _mm256_loadu_si256(p);
        }
        // Transpose bytes
        for k in 0..8 {
            let l = k * 2;
            ymm1[k * 2] = _mm256_unpacklo_epi8(ymm0[l], ymm0[l + 1]);
            ymm1[k * 2 + 1] = _mm256_unpackhi_epi8(ymm0[l], ymm0[l + 1]);
        }
        // Transpose words
        let mut l = 0;
        for k in 0..8 {
            ymm0[k * 2] = _mm256_unpacklo_epi16(ymm1[l], ymm1[l + 2]);
            ymm0[k * 2 + 1] = _mm256_unpackhi_epi16(ymm1[l], ymm1[l + 2]);
            l += 1;
            if k % 2 == 1 {
                l += 2;
            }
        }
        // Transpose double words
        l = 0;
        for k in 0..8 {
            ymm1[k * 2] = _mm256_unpacklo_epi32(ymm0[l], ymm0[l + 4]);
            ymm1[k * 2 + 1] = _mm256_unpackhi_epi32(ymm0[l], ymm0[l + 4]);
            l += 1;
            if k % 4 == 3 {
                l += 4;
            }
        }
        // Transpose quad words
        for k in 0..8 {
            ymm0[k * 2] = _mm256_unpacklo_epi64(ymm1[k], ymm1[k + 8]);
            ymm0[k * 2 + 1] = _mm256_unpackhi_epi64(ymm1[k], ymm1[k + 8]);
        }
        for k in 0..16 {
            ymm0[k] = _mm256_permute4x64_epi64(ymm0[k], 0xd8);
            ymm0[k] = _mm256_shuffle_epi8(ymm0[k], shmask);
        }
        // Store the result vectors
        for k in 0..16 {
            let p = dst.add(j + k * total_elements) as *mut __m256i;
            _mm256_storeu_si256(p, ymm0[k]);
        }
    }
}

/// AVX2 optimized shuffle for type sizes greater than 16 bytes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
#[target_feature(enable = "avx2")]
unsafe fn shuffle_tiled(
    vectorizable_elements: usize,
    total_elements: usize,
    ts: usize,
    src: *const u8,
    dst: *mut u8,
) {
    let mut ymm0: [__m256i; 16] = mem::zeroed();
    let mut ymm1: [__m256i; 16] = mem::zeroed();
    let vecs_rem = ts % SO128I;

    // Create the shuffle mask.
    // NOTE: The XMM/YMM 'set' intrinsics require the arguments to be ordered from
    // most to least significant (i.e., their order is reversed when compared to
    // loading the mask from an array).
    #[rustfmt::skip]
    let shmask = _mm256_set_epi8(
        0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04,
        0x0b, 0x03, 0x0a, 0x02, 0x09, 0x01, 0x08, 0x00,
        0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04,
        0x0b, 0x03, 0x0a, 0x02, 0x09, 0x01, 0x08, 0x00);

    for j in (0..vectorizable_elements).step_by(SO256I) {
        // Advance the offset into the type by the vector size (in bytes), unless this is
        // the initial iteration and the type size is not a multiple of the vector size.
        // In that case, only advance by the number of bytes necessary so that the number
        // of remaining bytes in the type will be a multiple of the vector size.
        let mut offset_into_type = 0;
        while offset_into_type < ts {
            // Fetch elements in groups of 512 bytes
            for k in 0..16 {
                let p0 = src.add(offset_into_type + (j + 2 * k + 1) * ts) as *const __m128i;
                let p1 = src.add(offset_into_type + (j + 2 * k) * ts) as *const __m128i;
                ymm0[k] = _mm256_loadu2_m128i(p0, p1);
            }
            // Transpose bytes
            for k in 0..8 {
                let l = 2 * k;
                ymm1[k * 2] = _mm256_unpacklo_epi8(ymm0[l], ymm0[l + 1]);
                ymm1[k * 2 + 1] = _mm256_unpackhi_epi8(ymm0[l], ymm0[l + 1]);
            }
            // Transpose words
            let mut l = 0;
            for k in 0..8 {
                ymm0[k * 2] = _mm256_unpacklo_epi16(ymm1[l], ymm1[l + 2]);
                ymm0[k * 2 + 1] = _mm256_unpackhi_epi16(ymm1[l], ymm1[l + 2]);
                l += 1;
                if k % 2 == 1 {
                    l += 2;
                }
            }
            // Transpose double words
            l = 0;
            for k in 0..8 {
                ymm1[k * 2] = _mm256_unpacklo_epi32(ymm0[l], ymm0[l + 4]);
                ymm1[k * 2 + 1] = _mm256_unpackhi_epi32(ymm0[l], ymm0[l + 4]);
                l += 1;
                if k % 4 == 3 {
                    l += 4;
                }
            }
            // Transpose quad words
            for k in 0..8 {
                ymm0[k * 2] = _mm256_unpacklo_epi64(ymm1[k], ymm1[k + 8]);
                ymm0[k * 2 + 1] = _mm256_unpackhi_epi64(ymm1[k], ymm1[k + 8]);
            }
            for k in 0..16 {
                ymm0[k] = _mm256_permute4x64_epi64(ymm0[k], 0xd8);
                ymm0[k] = _mm256_shuffle_epi8(ymm0[k], shmask);
            }
            // Store the result vectors
            for k in 0..16 {
                let p = dst.add(j + total_elements * (offset_into_type + k)) as *mut __m256i;
                _mm256_storeu_si256(p, ymm0[k]);
            }
            offset_into_type += if offset_into_type == 0 && vecs_rem > 0 {
                vecs_rem
            } else {
                SO128I
            };
        }
    }
}

// Gather-load based implementation.  Unfortunately, it's slower than the other one, probably
// because the _mm_i32gather_epi32 instruction is so slow.  On the plus side, it uses fewer
// registers than the other one.
//
// /// AVX2 optimized shuffle for 16-byte type sizes,
// #[allow(clippy::needless_range_loop)]   // I don't like this suggestion
// #[target_feature(enable = "avx2")]
// unsafe fn shuffle16(
//     vectorizable_elements: usize,
//     total_elements: usize,
//     src: *const u8,
//     dst: *mut u8)
// {
//     assert_eq!(vectorizable_elements % 4, 0);
//     assert_eq!(total_elements, vectorizable_elements, "TODO");
//
//     const TS: usize = 16;
//     const SO128I: usize = mem::size_of::<__m128i>();
//     let mut xmm: [__m128i; 4] = mem::zeroed();
//     let mut xmm1: [__m128i; 4] = mem::zeroed();
//
//     let vindex = _mm_set_epi32(3 * SO128I as i32, 2 * SO128I as i32, SO128I as i32, 0);
//     let shuf8 = _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
//
//     for i in 0..vectorizable_elements / 16 {
//         for j in (0..16).step_by(4) {
//             // Outline:
//             // Starting with data that looks like [0, 1, 2, 3, ... 255]
//             // for k in 0, 4, 8, 12
//             //   * gather-load into 4 __m128i registers that each look like 0123012301230123
//             //   * shuffle each so it looks like 0000111122223333
//             //   * Use i32 shuffles and blends to get four registers that look like
//             //     - 0000000000000000
//             //     - 1111111111111111
//             //     - 2222222222222222
//             //     - 3333333333333333
//             //   * Write to dst
//             // Tip:
//             //   View an __m128i variable in rust-gdb with "p/x *(&xmm1[k] as *const u8)@16"
//             for k in 0..4 {
//                 let p = src.add(i * 16 * TS + j + 4 * TS * k) as *const i32;
//                 xmm[k] = _mm_i32gather_epi32(p, vindex, 1);
//                 // xmm[0] should look like [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
//                 xmm[k] = _mm_shuffle_epi8(xmm[k], shuf8);
//                 // xmm[0] should look like [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51]
//             }
//             // This next step is logically a rotation, but we use shuffle instructions
//             xmm[1] = _mm_shuffle_epi32(xmm[1], 0x93);
//             xmm[2] = _mm_shuffle_epi32(xmm[2], 0x4e);
//             xmm[3] = _mm_shuffle_epi32(xmm[3], 0x39);
//             // xmm[0] should be unchanged, xmm[1] rotated 4 bytes right, xmm[2] 8 bytes right, etc
//
//             xmm1[0] = _mm_blend_epi32(xmm[0], xmm[1], 0b0010);
//             xmm1[0] = _mm_blend_epi32(xmm1[0], xmm[2], 0b0100);
//             xmm1[0] = _mm_blend_epi32(xmm1[0], xmm[3], 0b1000);
//             xmm1[1] = _mm_blend_epi32(xmm[0], xmm[1], 0b0100);
//             xmm1[1] = _mm_blend_epi32(xmm1[1], xmm[2], 0b1000);
//             xmm1[1] = _mm_blend_epi32(xmm1[1], xmm[3], 0b0001);
//             xmm1[2] = _mm_blend_epi32(xmm[0], xmm[1], 0b1000);
//             xmm1[2] = _mm_blend_epi32(xmm1[2], xmm[2], 0b0001);
//             xmm1[2] = _mm_blend_epi32(xmm1[2], xmm[3], 0b0010);
//             xmm1[3] = _mm_blend_epi32(xmm[0], xmm[1], 0b0001);
//             xmm1[3] = _mm_blend_epi32(xmm1[3], xmm[2], 0b0010);
//             xmm1[3] = _mm_blend_epi32(xmm1[3], xmm[3], 0b0100);
//
//             // Now to rotate the xmm1 registers, again using shuffle instructions
//             xmm1[1] = _mm_shuffle_epi32(xmm1[1], 0x39);
//             xmm1[2] = _mm_shuffle_epi32(xmm1[2], 0x4e);
//             xmm1[3] = _mm_shuffle_epi32(xmm1[3], 0x93);
//
//             // Now write the results
//             for k in 0..4 {
//                 let p = dst.add((i + (j + k) * vectorizable_elements / 16) * TS) as *mut __m128i;
//                 _mm_storeu_si128(p, xmm1[k]);
//             }
//         }
//     }
// }

pub unsafe fn shuffle(typesize: usize, len: usize, src: *const u8, dst: *mut u8) {
    let vectorized_chunk_size = typesize * mem::size_of::<__m256i>();
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
        crate::generic::shuffle(typesize, len, src, dst)
    }

    // If the buffer had any bytes at the end which couldn't be handled
    // by the vectorized implementations, use the non-optimized version
    // to finish them up.
    if vectorizable_bytes < len {
        crate::generic::shuffle_partial(typesize, vectorizable_bytes, len, src, dst);
    }
}

#[cfg(test)]
mod t {
    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                eprintln!("Skipping: AVX2 unavailable.");
                return;
            }
        };
    }

    mod shuffle {
        use rand::Rng;
        use rstest::rstest;

        #[rstest]
        #[case(16, 256)]
        #[case(16, 4096)]
        #[case(16, 4352)]
        fn compare(#[case] typesize: usize, #[case] len: usize) {
            require_avx2!();
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
            require_avx2!();
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
            require_avx2!();
            let typesize = 16;
            let len = 512;
            let src = (0..len).map(|i| (i % 256) as u8).collect::<Vec<u8>>();
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
