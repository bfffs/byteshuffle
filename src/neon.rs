//! NEON-optimized routines
use std::arch::aarch64::{
    uint16x8_t,
    uint32x4_t,
    uint64x2_t,
    uint8x16_t,
    vdupq_n_u8,
    vld1q_u8,
    vreinterpretq_u16_u8,
    vreinterpretq_u32_u8,
    vreinterpretq_u64_u8,
    vreinterpretq_u8_u16,
    vreinterpretq_u8_u32,
    vreinterpretq_u8_u64,
    vst1q_u8,
    vuzp1q_u8,
    vuzp2q_u8,
    vzip1q_u16,
    vzip1q_u32,
    vzip1q_u64,
    vzip1q_u8,
    vzip2q_u16,
    vzip2q_u32,
    vzip2q_u64,
    vzip2q_u8,
};

const SO128I: usize = std::mem::size_of::<uint8x16_t>();

/// NEON optimized shuffle for 2-byte type sizes
unsafe fn shuffle2(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 2;
    for j in (0..vectorizable_elements).step_by(SO128I) {
        let v0 = vld1q_u8(src.add(j * TS));
        let v1 = vld1q_u8(src.add(j * TS + SO128I));

        let l = vuzp1q_u8(v0, v1);
        let h = vuzp2q_u8(v0, v1);

        vst1q_u8(dst.add(j), l);
        vst1q_u8(dst.add(j + total_elements), h);
    }
}

/// NEON optimized shuffle for 16-byte type sizes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
unsafe fn shuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 16;
    let mut xmm0: [uint8x16_t; 16] = [vdupq_n_u8(0); 16];
    let mut xmm1: [uint8x16_t; 16] = [vdupq_n_u8(0); 16];

    for j in (0..vectorizable_elements).step_by(SO128I) {
        for k in 0..16 {
            xmm0[k] = vld1q_u8(src.add(j * TS + k * SO128I));
        }
        // Transpose bytes
        for k in 0..8 {
            let l = k * 2;
            xmm1[k * 2] = vzip1q_u8(xmm0[l], xmm0[l + 1]);
            xmm1[k * 2 + 1] = vzip2q_u8(xmm0[l], xmm0[l + 1]);
        }
        // Transpose words
        let mut l = 0;
        for k in 0..8 {
            let v0: uint16x8_t = vreinterpretq_u16_u8(xmm1[l]);
            let v1: uint16x8_t = vreinterpretq_u16_u8(xmm1[l + 2]);
            xmm0[k * 2] = vreinterpretq_u8_u16(vzip1q_u16(v0, v1));
            xmm0[k * 2 + 1] = vreinterpretq_u8_u16(vzip2q_u16(v0, v1));
            l += 1;
            if k % 2 == 1 {
                l += 2;
            }
        }
        // Transpose double words
        l = 0;
        for k in 0..8 {
            let v0: uint32x4_t = vreinterpretq_u32_u8(xmm0[l]);
            let v1: uint32x4_t = vreinterpretq_u32_u8(xmm0[l + 4]);
            xmm1[k * 2] = vreinterpretq_u8_u32(vzip1q_u32(v0, v1));
            xmm1[k * 2 + 1] = vreinterpretq_u8_u32(vzip2q_u32(v0, v1));
            l += 1;
            if k % 4 == 3 {
                l += 4;
            }
        }
        // Transpose quad words
        for k in 0..8 {
            let v0: uint64x2_t = vreinterpretq_u64_u8(xmm1[k]);
            let v1: uint64x2_t = vreinterpretq_u64_u8(xmm1[k + 8]);
            xmm0[k * 2] = vreinterpretq_u8_u64(vzip1q_u64(v0, v1));
            xmm0[k * 2 + 1] = vreinterpretq_u8_u64(vzip2q_u64(v0, v1));
        }
        // Store the result vectors
        let dst_for_jth_element = dst.add(j);
        for k in 0..16 {
            vst1q_u8(dst_for_jth_element.add(k * total_elements), xmm0[k]);
        }
    }
}

/// NEON optimized shuffle for type sizes larger than 16 bytes
#[allow(clippy::needless_range_loop)]
unsafe fn shuffle_tiled(
    vectorizable_elements: usize,
    total_elements: usize,
    ts: usize,
    src: *const u8,
    dst: *mut u8,
) {
    let mut xmm0: [uint8x16_t; 16] = [vdupq_n_u8(0); 16];
    let mut xmm1: [uint8x16_t; 16] = [vdupq_n_u8(0); 16];
    let vecs_rem = ts % SO128I;

    for j in (0..vectorizable_elements).step_by(SO128I) {
        let mut off_in_ty = 0;
        while off_in_ty < ts {
            // Fetch elements in groups of 256 bytes
            for k in 0..16 {
                xmm0[k] = vld1q_u8(src.add(off_in_ty + (j + k) * ts));
            }
            // Transpose bytes
            for k in 0..8 {
                let l = k * 2;
                xmm1[k * 2] = vzip1q_u8(xmm0[l], xmm0[l + 1]);
                xmm1[k * 2 + 1] = vzip2q_u8(xmm0[l], xmm0[l + 1]);
            }
            // Transpose words
            let mut l = 0;
            for k in 0..8 {
                let v0: uint16x8_t = vreinterpretq_u16_u8(xmm1[l]);
                let v1: uint16x8_t = vreinterpretq_u16_u8(xmm1[l + 2]);
                xmm0[k * 2] = vreinterpretq_u8_u16(vzip1q_u16(v0, v1));
                xmm0[k * 2 + 1] = vreinterpretq_u8_u16(vzip2q_u16(v0, v1));
                l += 1;
                if (k % 2) == 1 {
                    l += 2;
                }
            }
            // Transpose double words
            let mut l = 0;
            for k in 0..8 {
                let v0: uint32x4_t = vreinterpretq_u32_u8(xmm0[l]);
                let v1: uint32x4_t = vreinterpretq_u32_u8(xmm0[l + 4]);
                xmm1[k * 2] = vreinterpretq_u8_u32(vzip1q_u32(v0, v1));
                xmm1[k * 2 + 1] = vreinterpretq_u8_u32(vzip2q_u32(v0, v1));
                l += 1;
                if (k % 4) == 3 {
                    l += 4;
                }
            }
            // Transpose quad words
            for k in 0..8 {
                let v0: uint64x2_t = vreinterpretq_u64_u8(xmm1[k]);
                let v1: uint64x2_t = vreinterpretq_u64_u8(xmm1[k + 8]);
                xmm0[k * 2] = vreinterpretq_u8_u64(vzip1q_u64(v0, v1));
                xmm0[k * 2 + 1] = vreinterpretq_u8_u64(vzip2q_u64(v0, v1));
            }
            // Store the result vectors
            for k in 0..16 {
                vst1q_u8(dst.add(j + total_elements * (off_in_ty + k)), xmm0[k]);
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
    let vectorized_chunk_size = typesize * SO128I;
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
        return;
    }

    // If the buffer had any bytes at the end which couldn't be handled
    // by the vectorized implementations, use the non-optimized version
    // to finish them up.
    if vectorizable_bytes < len {
        crate::generic::shuffle_partial(typesize, vectorizable_bytes, len, src, dst);
    }
}

/// NEON optimized unshuffle for 2-byte type sizes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
unsafe fn unshuffle2(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 2;
    for i in (0..vectorizable_elements).step_by(SO128I) {
        let v0 = vld1q_u8(src.add(i));
        let v1 = vld1q_u8(src.add(i + total_elements));

        let l = vzip1q_u8(v0, v1);
        let h = vzip2q_u8(v0, v1);

        vst1q_u8(dst.add(i * TS), l);
        vst1q_u8(dst.add(i * TS + SO128I), h);
    }
}

/// NEON optimized unshuffle for 16-byte type sizes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
unsafe fn unshuffle16(
    vectorizable_elements: usize,
    total_elements: usize,
    src: *const u8,
    dst: *mut u8,
) {
    const TS: usize = 16;
    let mut xmm1: [uint8x16_t; 16] = [vdupq_n_u8(0); 16];
    let mut xmm2: [uint8x16_t; 16] = [vdupq_n_u8(0); 16];

    for i in (0..vectorizable_elements).step_by(SO128I) {
        // Load 16 elements (256 bytes) into 16 XMM registers.
        for j in 0..16 {
            xmm1[j] = vld1q_u8(src.add(i + j * total_elements));
        }
        // Shuffle bytes
        for j in 0..8 {
            xmm2[j] = vzip1q_u8(xmm1[j * 2], xmm1[j * 2 + 1]);
            xmm2[8 + j] = vzip2q_u8(xmm1[j * 2], xmm1[j * 2 + 1]);
        }
        // Shuffle 2-byte words
        for j in 0..8 {
            let v0 = vreinterpretq_u16_u8(xmm2[j * 2]);
            let v1 = vreinterpretq_u16_u8(xmm2[j * 2 + 1]);
            xmm1[j] = vreinterpretq_u8_u16(vzip1q_u16(v0, v1));
            xmm1[8 + j] = vreinterpretq_u8_u16(vzip2q_u16(v0, v1));
        }
        // Shuffle 4-byte dwords
        for j in 0..8 {
            let v0 = vreinterpretq_u32_u8(xmm1[j * 2]);
            let v1 = vreinterpretq_u32_u8(xmm1[j * 2 + 1]);
            xmm2[j] = vreinterpretq_u8_u32(vzip1q_u32(v0, v1));
            xmm2[8 + j] = vreinterpretq_u8_u32(vzip2q_u32(v0, v1));
        }
        // Shuffle 8-byte qwords
        for j in 0..8 {
            let v0 = vreinterpretq_u64_u8(xmm2[j * 2]);
            let v1 = vreinterpretq_u64_u8(xmm2[j * 2 + 1]);
            xmm1[j] = vreinterpretq_u8_u64(vzip1q_u64(v0, v1));
            xmm1[8 + j] = vreinterpretq_u8_u64(vzip2q_u64(v0, v1));
        }

        // Store the result vectors in proper order
        #[allow(clippy::erasing_op)]
        vst1q_u8(dst.add(i * TS + 0 * SO128I), xmm1[0]);
        #[allow(clippy::identity_op)]
        vst1q_u8(dst.add(i * TS + 1 * SO128I), xmm1[8]);
        vst1q_u8(dst.add(i * TS + 2 * SO128I), xmm1[4]);
        vst1q_u8(dst.add(i * TS + 3 * SO128I), xmm1[12]);
        vst1q_u8(dst.add(i * TS + 4 * SO128I), xmm1[2]);
        vst1q_u8(dst.add(i * TS + 5 * SO128I), xmm1[10]);
        vst1q_u8(dst.add(i * TS + 6 * SO128I), xmm1[6]);
        vst1q_u8(dst.add(i * TS + 7 * SO128I), xmm1[14]);
        vst1q_u8(dst.add(i * TS + 8 * SO128I), xmm1[1]);
        vst1q_u8(dst.add(i * TS + 9 * SO128I), xmm1[9]);
        vst1q_u8(dst.add(i * TS + 10 * SO128I), xmm1[5]);
        vst1q_u8(dst.add(i * TS + 11 * SO128I), xmm1[13]);
        vst1q_u8(dst.add(i * TS + 12 * SO128I), xmm1[3]);
        vst1q_u8(dst.add(i * TS + 13 * SO128I), xmm1[11]);
        vst1q_u8(dst.add(i * TS + 14 * SO128I), xmm1[7]);
        vst1q_u8(dst.add(i * TS + 15 * SO128I), xmm1[15]);
    }
}

/// NEON optimized unshuffle for type sizes larger than 16 bytes
#[allow(clippy::needless_range_loop)] // I don't like this suggestion
unsafe fn unshuffle_tiled(
    vectorizable_elements: usize,
    total_elements: usize,
    ts: usize,
    src: *const u8,
    dst: *mut u8,
) {
    let mut xmm1: [uint8x16_t; 16] = [vdupq_n_u8(0); 16];
    let mut xmm2: [uint8x16_t; 16] = [vdupq_n_u8(0); 16];
    let vecs_rem = ts % SO128I;

    let mut off_in_ty = 0;
    while off_in_ty < ts {
        for i in (0..vectorizable_elements).step_by(SO128I) {
            for j in 0..16 {
                xmm1[j] = vld1q_u8(src.add(i + total_elements * (off_in_ty + j)));
            }
            // Shuffle bytes
            for j in 0..8 {
                xmm2[j] = vzip1q_u8(xmm1[j * 2], xmm1[j * 2 + 1]);
                xmm2[8 + j] = vzip2q_u8(xmm1[j * 2], xmm1[j * 2 + 1]);
            }
            // Shuffle 2-byte words
            for j in 0..8 {
                let v0 = vreinterpretq_u16_u8(xmm2[j * 2]);
                let v1 = vreinterpretq_u16_u8(xmm2[j * 2 + 1]);
                xmm1[j] = vreinterpretq_u8_u16(vzip1q_u16(v0, v1));
                xmm1[8 + j] = vreinterpretq_u8_u16(vzip2q_u16(v0, v1));
            }
            // Shuffle 4-byte dwords
            for j in 0..8 {
                let v0 = vreinterpretq_u32_u8(xmm1[j * 2]);
                let v1 = vreinterpretq_u32_u8(xmm1[j * 2 + 1]);
                xmm2[j] = vreinterpretq_u8_u32(vzip1q_u32(v0, v1));
                xmm2[8 + j] = vreinterpretq_u8_u32(vzip2q_u32(v0, v1));
            }
            // Shuffle 8-byte qwords
            for j in 0..8 {
                let v0 = vreinterpretq_u64_u8(xmm2[j * 2]);
                let v1 = vreinterpretq_u64_u8(xmm2[j * 2 + 1]);
                xmm1[j] = vreinterpretq_u8_u64(vzip1q_u64(v0, v1));
                xmm1[8 + j] = vreinterpretq_u8_u64(vzip2q_u64(v0, v1));
            }
            // Store the result vectors in proper order
            #[allow(clippy::identity_op)]
            vst1q_u8(dst.add(off_in_ty + (i + 0) * ts), xmm1[0]);
            vst1q_u8(dst.add(off_in_ty + (i + 1) * ts), xmm1[8]);
            vst1q_u8(dst.add(off_in_ty + (i + 2) * ts), xmm1[4]);
            vst1q_u8(dst.add(off_in_ty + (i + 3) * ts), xmm1[12]);
            vst1q_u8(dst.add(off_in_ty + (i + 4) * ts), xmm1[2]);
            vst1q_u8(dst.add(off_in_ty + (i + 5) * ts), xmm1[10]);
            vst1q_u8(dst.add(off_in_ty + (i + 6) * ts), xmm1[6]);
            vst1q_u8(dst.add(off_in_ty + (i + 7) * ts), xmm1[14]);
            vst1q_u8(dst.add(off_in_ty + (i + 8) * ts), xmm1[1]);
            vst1q_u8(dst.add(off_in_ty + (i + 9) * ts), xmm1[9]);
            vst1q_u8(dst.add(off_in_ty + (i + 10) * ts), xmm1[5]);
            vst1q_u8(dst.add(off_in_ty + (i + 11) * ts), xmm1[13]);
            vst1q_u8(dst.add(off_in_ty + (i + 12) * ts), xmm1[3]);
            vst1q_u8(dst.add(off_in_ty + (i + 13) * ts), xmm1[11]);
            vst1q_u8(dst.add(off_in_ty + (i + 14) * ts), xmm1[7]);
            vst1q_u8(dst.add(off_in_ty + (i + 15) * ts), xmm1[15]);
        }
        off_in_ty += if off_in_ty == 0 && vecs_rem > 0 {
            vecs_rem
        } else {
            SO128I
        };
    }
}

pub unsafe fn unshuffle(typesize: usize, len: usize, src: *const u8, dst: *mut u8) {
    let vectorized_chunk_size = typesize * SO128I;
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
            let mut rng = rand::rng();

            let src = (0..len).map(|_| rng.random()).collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut neon_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::neon::shuffle(typesize, len, src.as_ptr(), neon_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, neon_dst);
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
        #[case(18, 288)]
        fn compare(#[case] typesize: usize, #[case] len: usize) {
            let mut rng = rand::rng();

            let src = (0..len).map(|_| rng.random()).collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut neon_dst = vec![0u8; len];
            unsafe {
                crate::generic::unshuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                crate::neon::unshuffle(typesize, len, src.as_ptr(), neon_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, neon_dst);
        }
    }
}
