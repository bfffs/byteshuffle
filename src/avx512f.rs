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
#[target_feature(enable = "avx512f")]
unsafe fn shuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8)
{
    todo!()
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


