//! AVX2 optimized routines
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as simd;
#[cfg(target_arch = "x86")]
use core::arch::x86 as simd;
use simd::{
    __m512i, _mm512_i32gather_epi32, _mm512_set_epi32, _mm256_set_epi32, _mm512_set_epi8,
    _mm512_shuffle_epi8, _mm512_i32scatter_epi64,
_mm512_permutexvar_epi32
};

use std::mem;

const SOI32: usize = mem::size_of::<i32>();
const SO512I: usize = mem::size_of::<__m512i>();

/// Helper that shuffles 8 32-bit ints
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn shuffle_8x4(zmm: __m512i) -> __m512i {
    // Sadly, there's no const constructor for __m512i
    let shuf8 = _mm512_set_epi8(
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
    );
    let shuf32 = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, );
    let zmm1 = _mm512_shuffle_epi8(zmm, shuf8);
    _mm512_permutexvar_epi32(shuf32, zmm1)
}

/// AVX512F optimized shuffle for 16-byte type sizes,
#[allow(clippy::needless_range_loop)]   // I don't like this suggestion
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn shuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8)
{
    debug_assert_eq!(vectorizable_elements % 4, 0);
    debug_assert_eq!(total_elements, vectorizable_elements, "TODO");

    const TS: usize = 16;

    let loadindex = _mm512_set_epi32(
        15 * TS as i32,
        14 * TS as i32,
        13 * TS as i32,
        12 * TS as i32,
        11 * TS as i32,
        10 * TS as i32,
        9 * TS as i32,
        8 * TS as i32,
        7 * TS as i32,
        6 * TS as i32,
        5 * TS as i32,
        4 * TS as i32,
        3 * TS as i32,
        2 * TS as i32,
        TS as i32,
        0
    );
    let storeindex = _mm256_set_epi32(
        (vectorizable_elements /4 * 3 * TS / 4 + SOI32 * 2) as i32,
        (vectorizable_elements /4 * 3 * TS / 4) as i32,
        (vectorizable_elements /4 * TS / 2 + SOI32 * 2) as i32,
        (vectorizable_elements /4 * TS / 2) as i32,
        (vectorizable_elements /4 * TS / 4 + SOI32 * 2) as i32,
        (vectorizable_elements /4 * TS / 4) as i32,
        (SOI32 * 2) as i32,
        0
    );

    for i in 0..(vectorizable_elements / (SO512I / SOI32)) {
        for j in 0..(TS / SOI32) {
            let p = src.add(i * SOI32 * SO512I + j * TS / SOI32) as *const u8;
            let mut zmm = _mm512_i32gather_epi32(loadindex, p, 1);
            // zmm should look like [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67...]
            zmm = shuffle_8x4(zmm);
            // zmm should look like [0, 16, 32, 48, 64, 80, ... 1, 17, 33, ...]
            let p = dst.add(i * (SO512I / SOI32) + j * vectorizable_elements * SO512I / TS);
            _mm512_i32scatter_epi64(p, storeindex, zmm, 1);
        }
    }
}

/// AVX512F optimized shuffle for type sizes of at least 16 bytes
#[allow(clippy::needless_range_loop)]   // I don't like this suggestion
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn shuffle_tiled(
    vectorizable_elements: usize,
    total_elements: usize,
    ts: usize,
    src: *const u8,
    dst: *mut u8)
{
    debug_assert_eq!(vectorizable_elements % 4, 0);
    debug_assert_eq!(total_elements, vectorizable_elements, "TODO");

    let loadindex = _mm512_set_epi32(
        15 * ts as i32,
        14 * ts as i32,
        13 * ts as i32,
        12 * ts as i32,
        11 * ts as i32,
        10 * ts as i32,
        9 * ts as i32,
        8 * ts as i32,
        7 * ts as i32,
        6 * ts as i32,
        5 * ts as i32,
        4 * ts as i32,
        3 * ts as i32,
        2 * ts as i32,
        ts as i32,
        0
    );
    let storeindex = _mm256_set_epi32(
        (vectorizable_elements /4 * 3 * 16 / 4 + SOI32 * 2) as i32,
        (vectorizable_elements /4 * 3 * 16 / 4) as i32,
        (vectorizable_elements /4 * 16 / 2 + SOI32 * 2) as i32,
        (vectorizable_elements /4 * 16 / 2) as i32,
        (vectorizable_elements /4 * 16 / 4 + SOI32 * 2) as i32,
        (vectorizable_elements /4 * 16 / 4) as i32,
        (SOI32 * 2) as i32,
        0
    );

    for i in 0..(vectorizable_elements / (SO512I / SOI32)) {
        for j in 0..(ts / SOI32) {
            let p = src.add(i * SOI32 * SO512I + j * 16 / SOI32) as *const u8;
            let mut zmm = _mm512_i32gather_epi32(loadindex, p, 1);
            // zmm should look like [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67...]
            zmm = shuffle_8x4(zmm);
            // zmm should look like [0, 16, 32, 48, 64, 80, ... 1, 17, 33, ...]
            let p = dst.add(i * (SO512I / SOI32) + j * vectorizable_elements * 4);
            _mm512_i32scatter_epi64(p, storeindex, zmm, 1);
        }
        // Get remainders using byte loads
        // TODO: consider doing 16-bit loads, if ts%4 >= 2
        for k in (ts - ts % SOI32)..ts {
            for l in 0..(SO512I / SOI32) {
                *dst.add(l + k * vectorizable_elements) = *src.add(k + l * ts);

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
    let vectorized_chunk_size = typesize * SO512I / 4 ;
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
    } else if typesize > 16 {
        shuffle_tiled(vectorizable_elements, total_elements, typesize, src, dst);
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
    macro_rules! require_avx512f {
        () => {
            if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw")
            {
                eprintln!("Skipping: AVX512F unavailable.");
                return;
            }
        }
    }

    mod shuffle_8x4 {
        use rstest::rstest;
        use super::super::{shuffle_8x4};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64 as simd;
        #[cfg(target_arch = "x86")]
        use core::arch::x86 as simd;
        use simd::{_mm512_load_si512, _mm512_store_si512};

        #[rstest]
        fn t() {
            require_avx512f!();

            let input = vec![
                0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51,
                64, 65, 66, 67, 80, 81, 82, 83, 96, 97, 98, 99, 112, 113, 114, 115,
                128, 129, 130, 131, 144, 145, 146, 147, 160, 161, 162, 163, 176, 177, 178, 179,
                192, 193, 194, 195, 208, 209, 210, 211, 224, 225, 226, 227, 240, 241, 242, 243];
            let mut want = vec![0; 64];
            let mut actual = vec![0; 64];
            unsafe {
                let x = _mm512_load_si512(input.as_ptr() as *const i32);
                let y = shuffle_8x4(x);
                crate::generic::shuffle(4, input.len(), input.as_ptr(), want.as_mut_ptr());
                _mm512_store_si512(actual.as_mut_ptr() as *mut i32, y);
            }
            //let want = _mm512_set_epi8(0, 16, 32, 48, 60, 80, 96, 112, 1, 17, 33, 49, 65, 81, 97, 113, 2, 18, 34, 50, 66, 82, 98, 114, 3, 19, 35, 51, 57, 83, 99, 115);
            assert_eq!(want, actual);
        }
    }

    mod shuffle {
        use rand::Rng;
        use rstest::rstest;

        #[rstest]
        #[case(16, 256)]
        #[case(16, 512)]
        #[case(16, 4096)]
        #[case(16, 4352)]
        #[case(16, 65536)]
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

        // This test is redundant with the randomly generated one, but easier to debug.
        #[rstest]
        fn compare16x256() {
            require_avx512f!();
            let typesize = 16;
            let len = 256;
            let src = (0..len)
                .map(|i| (i % 256) as u8)
                .collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::avx512f::shuffle(typesize, len, src.as_ptr(), sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }

        #[rstest]
        fn compare16x272() {
            require_avx512f!();
            let typesize = 16;
            let len = 272;
            let src = (0..len)
                .map(|i| (i % 256) as u8)
                .collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::avx512f::shuffle(typesize, len, src.as_ptr(), sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }

        #[rstest]
        fn compare18x288() {
            require_avx512f!();
            let typesize = 18;
            let len = 288;
            let src = (0..len)
                .map(|i| (i % 256) as u8)
                .collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut sse2_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::avx512f::shuffle(typesize, len, src.as_ptr(), sse2_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, sse2_dst);
        }

        #[rstest]
        fn compare20x320() {
            require_avx512f!();
            let typesize = 20;
            let len = 320;
            let src = (0..len)
                .map(|i| (i % 256) as u8)
                .collect::<Vec<u8>>();
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

