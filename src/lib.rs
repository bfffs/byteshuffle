#![feature(avx512_target_feature)]
#![feature(stdsimd)]

use std::mem;

use cfg_if::cfg_if;
use ctor::ctor;

mod avx2;
mod avx512f;
mod generic;
mod sse2;

static mut SHUFFLE: unsafe fn(usize, usize, *const u8, *mut u8) = generic::shuffle;

#[ctor]
fn select_implementation() {
    // Safe because ctor guarantees only one writer at a time
    cfg_if! {
        if #[cfg(any(target_arch = "x86_64", target_arch = "x86"))] {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"){
                unsafe { SHUFFLE = avx512f::shuffle; }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { SHUFFLE = avx2::shuffle; }
            } else if is_x86_feature_detected!("sse2") {
                unsafe { SHUFFLE = sse2::shuffle; }
            } else {
                unsafe { SHUFFLE = generic::shuffle; }
            }
        } else {
            unsafe { SHUFFLE = generic::shuffle; }
        }
    }
}

/// Shuffle an array of fixed-size objects.
pub fn shuffle<T: Copy>(src: &[T], dst: &mut [T]) {
    assert_eq!(src.len(), dst.len());
    let ts = mem::size_of::<T>();
    assert!(ts > 1, "No point shuffling plain [u8]");
    // Safe because of the first assertion.
    unsafe {
        SHUFFLE(ts, src.len() * ts,
            src.as_ptr() as *const u8, dst.as_ptr() as *mut u8);
    }
}

/// Shuffle a byte array whose contents are known to approximately repeat with
/// period `typesize`.
pub fn shuffle_bytes(typesize: usize, src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    // Safe because of the first assertion.
    unsafe {
        SHUFFLE(typesize, src.len(),
            src.as_ptr() as *const u8, dst.as_ptr() as *mut u8);
    }
}

/// Unshuffle an array of fixed-size objects.
pub fn unshuffle<T: Copy>(src: &[T], dst: &mut [T]) {
    assert_eq!(src.len(), dst.len());
    let ts = mem::size_of::<T>();
    assert!(ts > 1, "No point shuffling plain [u8]");
    // Safe because of the first assertion.
    unsafe {
        generic::unshuffle(ts, src.len() * ts,
            src.as_ptr() as *const u8, dst.as_ptr() as *mut u8);
    }
}

pub fn unshuffle_bytes(typesize: usize, src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    unsafe {
        generic::unshuffle(typesize, src.len(),
            src.as_ptr() as *const u8, dst.as_ptr() as *mut u8);
    }
}

#[cfg(test)]
mod t {
    use super::*;

    /// Test the type-based shuffle API
    mod shuffle {
        use super::*;

        // Two elements of two bytes each
        #[test]
        fn twobytwo() {
            let src = [0x1234u16, 0x5678u16];
            let mut dst = [0u16; 2];
            shuffle(&src[..], &mut dst[..]);
            assert_eq!(dst, &[0x7834u16, 0x5612][..]);
        }

        // Four elements of four bytes each
        #[test]
        fn fourbyfour() {
            let src = [0x11223344u32, 0x55667788, 0x99aabbcc, 0xddeeff00];
            let mut dst = [0u32; 4];
            shuffle(&src[..], &mut dst[..]);
            assert_eq!(dst, &[0x00cc8844, 0xffbb7733, 0xeeaa6622, 0xdd995511][..]);
        }
    }

    mod shuffle_bytes {
        use rand::Rng;
        use rstest::rstest;

        /// Compare optimized results against generic results
        #[rstest]
        #[case::sse2(crate::sse2::shuffle, is_x86_feature_detected!("sse2"))]
        #[case::avx2(crate::avx2::shuffle, is_x86_feature_detected!("avx2"))]
        #[case::avx512f(crate::avx512f::shuffle,
                        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
                        )]
        fn compare(
            #[values(2, 4, 8, 16, 18, 32, 36, 43, 47)]
            typesize: usize,
            #[values(64, 256, 1024, 4096)]
            len: usize,
            #[case]
            f: unsafe fn(usize, usize, *const u8, *mut u8),
            #[case]
            has_feature: bool
            )
        {
            if !has_feature {
                eprintln!("Skipping: CPU feature unavailable.");
                return;
            }

            let mut rng = rand::thread_rng();

            let src = (0..len).map(|_| rng.gen()).collect::<Vec<u8>>();
            let mut generic_dst = vec![0u8; len];
            let mut opt_dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                f(typesize, len, src.as_ptr(), opt_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, opt_dst);
        }
    }

    /// Test the type-based shuffle API
    mod unshuffle {
        use super::*;

        // Two elements of two bytes each
        #[test]
        fn twobytwo() {
            let src = [0x7834u16, 0x5612];
            let mut dst = [0u16; 2];
            unshuffle(&src[..], &mut dst[..]);
            assert_eq!(dst, &[0x1234u16, 0x5678u16][..]);
        }

        // Four elements of four bytes each
        #[test]
        fn fourbyfour() {
            let src = [0x00cc8844, 0xffbb7733, 0xeeaa6622, 0xdd995511];
            let mut dst = [0u32; 4];
            unshuffle(&src[..], &mut dst[..]);
            assert_eq!(dst, &[0x11223344u32, 0x55667788, 0x99aabbcc, 0xddeeff00][..]);
        }
    }
}
