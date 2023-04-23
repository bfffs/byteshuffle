//! AVX2 optimized routines
#[cfg(target_arch = "x86")]
use core::arch::x86 as simd;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as simd;
use std::mem;

use simd::{
    __m128i,
    __m512i,
    _mm256_set_epi32,
    _mm512_i32gather_epi32,
    _mm512_i32scatter_epi64,
    _mm512_loadu_si512,
    _mm512_permutex2var_epi64,
    _mm512_set_epi32,
    _mm512_set_epi64,
    _mm512_set_epi8,
    _mm512_shuffle_epi8,
    _mm_set_epi8,
    _mm_storeu_si128,
    _mm512_storeu_si512,
    _mm512_unpackhi_epi64,
    _mm512_unpackhi_epi32,
    _mm512_unpackhi_epi16,
    _mm512_unpackhi_epi8,
    _mm512_unpacklo_epi64,
    _mm512_unpacklo_epi32,
    _mm512_unpacklo_epi16,
    _mm512_unpacklo_epi8,
_mm512_permutexvar_epi32,
    _mm512_permutexvar_epi8
};

const SOI32: usize = mem::size_of::<i32>();
const SO512I: usize = mem::size_of::<__m512i>();

/// Helper that shuffles 8 32-bit ints
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn shuffle_8x4(zmm: __m512i) -> __m512i {
    // Sadly, there's no const constructor for __m512i
    #[rustfmt::skip]
    let shuf8 = _mm512_set_epi8(
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
    );
    let shuf32 = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
    let zmm1 = _mm512_shuffle_epi8(zmm, shuf8);
    _mm512_permutexvar_epi32(shuf32, zmm1)
}

/// AVX-512F optimized shuffle for 2-byte type sizes,
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn shuffle2(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 2;
    let mut zmm0: [__m512i; 2] = mem::zeroed();
    let mut zmm1: [__m512i; 2] = mem::zeroed();
    #[rustfmt::skip]
    let shuf8 = _mm512_set_epi8(
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
    );
    let idx0 = _mm512_set_epi64(0xe, 0xc, 0xa, 8, 6, 4, 2, 0);
    let idx1 = _mm512_set_epi64(0xf, 0xd, 0xb, 9, 7, 5, 3, 1);

    for j in (0..vectorizable_elements).step_by(SO512I) {
        // Fetch 64 elements (128 bytes) then shuffle things into place.
        for k in 0..2 {
            let p = src.add(j * TS + k * SO512I) as *const i32;
            zmm0[k] = _mm512_loadu_si512(p);
            // Shuffle within 128-bit lanes
            zmm0[k] = _mm512_shuffle_epi8(zmm0[k], shuf8);
        }
        // Permute 64-bit elements between lanes of two registers
        zmm1[0] = _mm512_permutex2var_epi64(zmm0[0], idx0, zmm0[1]);
        zmm1[1] = _mm512_permutex2var_epi64(zmm0[0], idx1, zmm0[1]);

        // Store the result vectors
        for k in 0..2 {
            let p = dst.add(j + k * total_elements) as *mut i32;
            _mm512_storeu_si512(p, zmm1[k]);
        }
    }
}

/// AVX512F optimized shuffle for 16-byte type sizes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
#[target_feature(enable = "avx512vbmi")]
unsafe fn shuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 16;
    let mut zmm0: [__m512i; TS] = mem::zeroed();
    let mut zmm1: [__m512i; TS] = mem::zeroed();

    #[rustfmt::skip]
    let shmask = _mm512_set_epi8(
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
    let shuf32 = _mm512_set_epi32(
        15, 11, 7, 3,
        14, 10, 6, 2,
        13, 9, 5, 1,
        12, 8, 4, 0,
        );
    let shuf8 = _mm512_set_epi8(
        63, 47, 31, 15,
        62, 46, 30, 14,
        61, 45, 29, 13,
        60, 44, 28, 12,
        59, 43, 27, 11,
        58, 42, 26, 10,
        57, 41, 25, 9,
        56, 40, 24, 8,
        55, 39, 23, 7,
        54, 38, 22, 6,
        53, 37, 21, 5,
        52, 36, 20, 4,
        51, 35, 19, 3,
        50, 34, 18, 2,
        49, 33, 17, 1,
        48, 32, 16, 0,
        );

    for j in (0..vectorizable_elements).step_by(SO512I) {
        for k in 0..TS {
            let p = src.add(j * TS + k * SO512I) as *const i32;
            zmm0[k] = _mm512_loadu_si512(p);
        }
        for k in 0..(TS/2) {
            zmm1[k * 2] = _mm512_unpacklo_epi8(zmm0[k * 2], zmm0[k * 2 + 1]);
            zmm1[k * 2 + 1] = _mm512_unpackhi_epi8(zmm0[k * 2], zmm0[k * 2 + 1]);
        }

        let mut l = 0;
        for k in 0..(TS/2) {
            zmm0[k * 2] = _mm512_unpacklo_epi16(zmm1[l], zmm1[l + 2]);
            zmm0[k * 2 + 1] = _mm512_unpackhi_epi16(zmm1[l], zmm1[l + 2]);
            l += 1;
            if k % 2 == 1 {
                l += 2;
            }
        }

        l = 0;
        for k in 0..(TS/2) {
            zmm1[k * 2] = _mm512_unpacklo_epi32(zmm0[l], zmm0[l + 4]);
            zmm1[k * 2 + 1] = _mm512_unpackhi_epi32(zmm0[l], zmm0[l + 4]);
            l += 1;
            if k % 4 == 3 {
                l += 4;
            }
        }

        for k in 0..(TS / 2) {
            zmm0[k * 2] = _mm512_unpacklo_epi64(zmm1[k], zmm1[k + 8]);
            zmm0[k * 2 + 1] = _mm512_unpackhi_epi64(zmm1[k], zmm1[k + 8]);
        }

        for k in 0..TS {
            // NB: these two steps can be replaced by _mm512_permutexvar_epi8 with AVX512-VBMI
            zmm0[k] = _mm512_permutexvar_epi8(shuf8, zmm0[k]);
            // zmm1[k] = _mm512_permutexvar_epi32(shuf32, zmm0[k]);
            // zmm0[k] = _mm512_shuffle_epi8(zmm1[k], shmask);
        }

        //TODO: shuffle 128 and 256 bit lanes
        for k in 0..TS {
            let p = dst.add(j + k * total_elements) as *mut i32;
            _mm512_storeu_si512(p, zmm0[k]);
        }
    }
}

/// AVX512F optimized shuffle for type sizes of at least 16 bytes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn shuffle_sg(
    vectorizable_elements: usize,
    total_elements: usize,
    ts: usize,
    src: *const u8,
    dst: *mut u8,
) {
    debug_assert_eq!(vectorizable_elements % 4, 0);

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
        0,
    );
    let storeindex = _mm256_set_epi32(
        (3 * total_elements + SOI32 * 2) as i32,
        (3 * total_elements) as i32,
        (2 * total_elements + SOI32 * 2) as i32,
        (2 * total_elements) as i32,
        (total_elements + SOI32 * 2) as i32,
        total_elements as i32,
        (SOI32 * 2) as i32,
        0,
    );

    for i in 0..(vectorizable_elements / 16) {
        for j in 0..(ts / SOI32) {
            let p = src.add(i * 16 * ts + j * 16 / SOI32) as *const u8;
            let mut zmm = _mm512_i32gather_epi32(loadindex, p, 1);
            // zmm should look like [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67...]
            zmm = shuffle_8x4(zmm);
            // zmm should look like [0, 16, 32, 48, 64, 80, ... 1, 17, 33, ...]
            let p = dst.add(i * 16 + j * total_elements * 4);
            _mm512_i32scatter_epi64(p, storeindex, zmm, 1);
        }
    }
    // Get remainders using byte loads
    // TODO: consider doing 16-bit loads, if ts%4 >= 2
    for k in (ts - ts % SOI32)..ts {
        let vectorizable_spill_elements = total_elements - total_elements % 16;
        for l in (0..vectorizable_spill_elements).step_by(16) {
            let xmm = _mm_set_epi8(
                *src.add(k + (l + 15) * ts) as i8,
                *src.add(k + (l + 14) * ts) as i8,
                *src.add(k + (l + 13) * ts) as i8,
                *src.add(k + (l + 12) * ts) as i8,
                *src.add(k + (l + 11) * ts) as i8,
                *src.add(k + (l + 10) * ts) as i8,
                *src.add(k + (l + 9) * ts) as i8,
                *src.add(k + (l + 8) * ts) as i8,
                *src.add(k + (l + 7) * ts) as i8,
                *src.add(k + (l + 6) * ts) as i8,
                *src.add(k + (l + 5) * ts) as i8,
                *src.add(k + (l + 4) * ts) as i8,
                *src.add(k + (l + 3) * ts) as i8,
                *src.add(k + (l + 2) * ts) as i8,
                *src.add(k + (l + 1) * ts) as i8,
                *src.add(k + (l + 0) * ts) as i8,
            );
            _mm_storeu_si128(dst.add(l + k * total_elements) as *mut __m128i, xmm);
        }
        for l in vectorizable_spill_elements..total_elements {
            *dst.add(l + k * total_elements) = *src.add(k + l * ts);
        }
    }
}

pub unsafe fn shuffle(typesize: usize, len: usize, src: *const u8, dst: *mut u8) {
    let vectorized_chunk_size = typesize * SO512I;
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
    } else if typesize == 16{
        shuffle16(vectorizable_elements, total_elements, src, dst);
    } else {
        // XXX shuffle_sg can actually operate on vectorized_chunk size of typesize * SO512I / 4
        shuffle_sg(vectorizable_elements, total_elements, typesize, src, dst);
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
    macro_rules! require_avx512f {
        () => {
            if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
                eprintln!("Skipping: AVX512F unavailable.");
                return;
            }
        };
    }

    mod shuffle_8x4 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86 as simd;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64 as simd;

        use rstest::rstest;
        use simd::{_mm512_loadu_si512, _mm512_storeu_si512};

        use super::super::shuffle_8x4;

        #[rstest]
        fn t() {
            require_avx512f!();

            #[rustfmt::skip]
            let input = vec![
                0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51,
                64, 65, 66, 67, 80, 81, 82, 83, 96, 97, 98, 99, 112, 113, 114, 115,
                128, 129, 130, 131, 144, 145, 146, 147, 160, 161, 162, 163, 176, 177, 178, 179,
                192, 193, 194, 195, 208, 209, 210, 211, 224, 225, 226, 227, 240, 241, 242, 243];
            let mut want = vec![0; 64];
            let mut actual = vec![0; 64];
            unsafe {
                let x = _mm512_loadu_si512(input.as_ptr() as *const i32);
                let y = shuffle_8x4(x);
                crate::generic::shuffle(4, input.len(), input.as_ptr(), want.as_mut_ptr());
                _mm512_storeu_si512(actual.as_mut_ptr() as *mut i32, y);
            }
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
        fn compare16x1024() {
            require_avx512f!();
            let typesize = 16;
            let len = 1024;
            let src = (0..len).map(|i| i.min(255) as u8).collect::<Vec<u8>>();
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
            let src = (0..len).map(|i| (i % 256) as u8).collect::<Vec<u8>>();
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
            let src = (0..len).map(|i| (i % 256) as u8).collect::<Vec<u8>>();
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
            let src = (0..len).map(|i| (i % 256) as u8).collect::<Vec<u8>>();
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
