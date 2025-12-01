//! SIMD-accelerated byte shuffle/unshuffle routines
//!
//! The byte-shuffle is a very efficient way to improve the compressibility of data that consists
//! of an array of fixed-size objects.  It rearranges the array in order to group all elements'
//! least significant bytes together, most-significant bytes together, and everything in between.
//! Since real applications' arrays often contain consecutive elements that are closely correlated
//! with each other, this filter frequently results in lengthy continuous runs of identical bytes.
//! Such runs are highly compressible by general-purpose compression libraries like gzip, lz4, etc.
//!
//! The [blosc](https://www.blosc.org) project was the original inspiration for this library.
//! Blosc is a C library intended primarily for HPC users, and it implements a shuffle filter,
//! among many other things.  This crate is a clean reimplementation of Blosc's shuffle filter.
//!
//! # Examples
//!
//! Typical use: a byte array consists of an arithmetic sequence.  Shuffle it, compress it,
//! decompress it, and then unshuffle it.
//! ```
//! # use byteshuffle::*;
//! const IN: [u8; 8] = [0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04];
//! let mut shuffled = [0u8; 8];
//! shuffle_bytes(2, &IN, &mut shuffled);
//! assert_ne!(IN, shuffled);
//! // In normal use, you would now serialize `shuffled`.  Then compress it, and later decompress
//! // and deserialize it.  Then unshuffle like below.
//! let mut unshuffled = [0u8; 8];
//! unshuffle_bytes(2, &shuffled, &mut unshuffled);
//! assert_eq!(IN, unshuffled);
//! ```
use std::{mem, str::FromStr};

use cfg_if::cfg_if;
use ctor::ctor;

mod avx2;
mod avx512f;
mod generic;
mod sse2;

static mut SHUFFLE: unsafe fn(usize, usize, *const u8, *mut u8) = generic::shuffle;

/// Explicitly specify an instruction set to use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SimdImpl {
    /// Automatically determine the implementation to use
    Auto,
    /// Don't use any SIMD accelerations
    Generic,
    /// Use the SSE2 instruction set
    Sse2,
    /// Use the AVX2 instruction set
    Avx2,
    /// Use the AVX512F + AVX512BW instruction sets
    Avx512F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseSimdImplErr;

impl FromStr for SimdImpl {
    type Err = ParseSimdImplErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(SimdImpl::Auto),
            "generic" => Ok(SimdImpl::Generic),
            "sse2" => Ok(SimdImpl::Sse2),
            "avx2" => Ok(SimdImpl::Avx2),
            "avx512f" => Ok(SimdImpl::Avx512F),
            _ => Err(ParseSimdImplErr),
        }
    }
}

#[ctor]
fn select_implementation_ctor() {
    // Safe because we're single-threaded before main
    unsafe { select_implementation(SimdImpl::Auto) }
}

/// Force the use of a particular CPU instruction set, globally.
///
/// The default behavior is to automatically select, which should normally work best.  But this
/// function may be used to force a lower instruction set for test and benchmarking purposes, or if
/// one of the higher level optimizations does not work well.
///
/// # Safety
///
/// May not be called concurrently with any other function in this library.
pub unsafe fn select_implementation(impl_: SimdImpl) {
    // Safe because ctor guarantees only one writer at a time
    cfg_if! {
        if #[cfg(any(target_arch = "x86_64", target_arch = "x86"))] {
            match impl_ {
                SimdImpl::Auto => {
                    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"){
                        unsafe { SHUFFLE = avx512f::shuffle; }
                    } else if is_x86_feature_detected!("avx2") {
                        unsafe { SHUFFLE = avx2::shuffle; }
                    } else if is_x86_feature_detected!("sse2") {
                        unsafe { SHUFFLE = sse2::shuffle; }
                    } else {
                        unsafe { SHUFFLE = generic::shuffle; }
                    }
                },
                SimdImpl::Generic => unsafe { SHUFFLE = generic::shuffle; },
                SimdImpl::Sse2 => unsafe { SHUFFLE = sse2::shuffle; },
                SimdImpl::Avx2 => unsafe { SHUFFLE = avx2::shuffle; },
                SimdImpl::Avx512F => unsafe { SHUFFLE = avx512f::shuffle; },
            }
        } else {
            unsafe { SHUFFLE = generic::shuffle; }
        }
    }
}

/// Shuffle an array of fixed-size objects.
///
/// The result will be a shuffled byte array.  Be aware that this byte array will usually not be
/// portable.  It can only be unshuffled by the same computer architecture as the one that shuffled
/// it, and sometimes only by the same compiler version.  If portability is desired, then first
/// serialize the data and then use [`shuffle_bytes`].
///
/// # Examples
/// ```
/// # use byteshuffle::*;
/// const IN: [u16; 4] = [0x01, 0x02, 0x03, 0x04];
/// let mut out = [0u8; 8];
/// shuffle(&IN, &mut out);
#[cfg_attr(
    target_endian = "little",
    doc = "assert_eq!(out, [0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00]);"
)]
#[cfg_attr(
    target_endian = "big",
    doc = "assert_eq!(out, [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04]);"
)]
/// ```
pub fn shuffle<T: Copy>(src: &[T], dst: &mut [u8]) {
    let ts = mem::size_of::<T>();
    assert_eq!(mem::size_of_val(src), dst.len());
    assert!(ts > 1, "No point shuffling plain [u8]");
    // Safe because of the first assertion.
    unsafe {
        SHUFFLE(
            ts,
            mem::size_of_val(src),
            src.as_ptr() as *const u8,
            dst.as_ptr() as *mut u8,
        );
    }
}

/// Shuffle a byte array whose contents are known to approximately repeat with
/// period `typesize`.
///
/// # Examples
/// ```
/// # use byteshuffle::*;
/// const IN: [u8; 8] = [0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04];
/// let mut out = [0u8; 8];
/// shuffle_bytes(2, &IN, &mut out);
/// assert_eq!(out, [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04]);
/// ```
pub fn shuffle_bytes(typesize: usize, src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    // Safe because of the first assertion.
    unsafe {
        SHUFFLE(typesize, src.len(), src.as_ptr(), dst.as_ptr() as *mut u8);
    }
}

/// Unshuffle an array of fixed-size objects.
///
/// # Safety
///
/// The `src` must've originally been produced by [`shuffle`], by a program that uses the exact
/// same byte layout as this one.  That means the same wordsize, same endianness, and may even
/// require the same compiler version.  If you can't guarantee those conditions, then serialize and
/// use [`shuffle_bytes`]/[`unshuffle_bytes`] instead.
///
/// # Examples
/// ```
/// # use byteshuffle::*;
/// const IN: [u16; 4] = [0x01, 0x02, 0x03, 0x04];
/// let mut shuffled = [0u8; 8];
/// shuffle(&IN, &mut shuffled);
/// let mut out = [0u16; 4];
/// unsafe { unshuffle(&shuffled, &mut out) };
/// assert_eq!(IN, out);
/// ```
pub unsafe fn unshuffle<T: Copy>(src: &[u8], dst: &mut [T]) {
    let ts = mem::size_of::<T>();
    assert_eq!(src.len(), mem::size_of_val(dst));
    assert!(ts > 1, "No point shuffling plain [u8]");
    // Safe because of the first assertion.
    unsafe {
        generic::unshuffle(
            ts,
            mem::size_of_val(dst),
            src.as_ptr(),
            dst.as_ptr() as *mut u8,
        );
    }
}

/// Unshuffle a byte array whose contents are known to approximately repeat with
/// period `typesize`.
///
/// # Examples
/// ```
/// # use byteshuffle::*;
/// const IN: [u8; 8] = [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04];
/// let mut out = [0u8; 8];
/// unshuffle_bytes(2, &IN, &mut out);
/// assert_eq!(out, [0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04]);
/// ```
pub fn unshuffle_bytes(typesize: usize, src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    unsafe {
        generic::unshuffle(typesize, src.len(), src.as_ptr(), dst.as_ptr() as *mut u8);
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
            let mut dst = [0u8; 4];
            shuffle(&src[..], &mut dst[..]);
            cfg_if! {
                if #[cfg(target_endian = "big")] {
                    let expected = [0x78u8, 0x34, 0x56, 0x12];
                } else {
                    let expected = [0x34, 0x78u8, 0x12, 0x56];
                }
            }
            assert_eq!(dst, &expected[..]);
        }

        // Four elements of four bytes each
        #[test]
        fn fourbyfour() {
            let src = [0x11223344u32, 0x55667788, 0x99aabbcc, 0xddeeff00];
            let mut dst = [0u8; 16];
            shuffle(&src[..], &mut dst[..]);
            cfg_if! {
                if #[cfg(target_endian = "big")] {
                    let expected = [0x00u8, 0xcc, 0x88, 0x44, 0xff, 0xbb, 0x77, 0x33,
                       0xee, 0xaa, 0x66, 0x22, 0xdd, 0x99, 0x55, 0x11];
                } else {
                    let expected = [0x44, 0x88, 0xcc, 0x00, 0x33, 0x77, 0xbb, 0xff,
                       0x22, 0x66, 0xaa, 0xee, 0x11, 0x55, 0x99, 0xdd];
                }
            }
            assert_eq!(dst, &expected[..]);
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
            #[values(2, 4, 8, 16, 18, 32, 36, 43, 47)] typesize: usize,
            #[values(64, 65, 256, 258, 1024, 1028, 4096, 4112)] len: usize,
            #[case] f: unsafe fn(usize, usize, *const u8, *mut u8),
            #[case] has_feature: bool,
        ) {
            if !has_feature {
                eprintln!("Skipping: CPU feature unavailable.");
                return;
            }

            let mut rng = rand::rng();

            let src = (0..len).map(|_| rng.random()).collect::<Vec<u8>>();
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
        use rand::Rng;
        use rstest::rstest;

        use super::*;

        // Two elements of two bytes each
        #[test]
        fn twobytwo() {
            cfg_if! {
                if #[cfg(target_endian = "big")] {
                    let src = [0x78u8, 0x34, 0x56, 0x12];
                } else {
                    let src = [0x34, 0x78u8, 0x12, 0x56];
                }
            }
            let mut dst = [0u16; 2];
            unsafe { unshuffle(&src[..], &mut dst[..]) };
            assert_eq!(dst, &[0x1234u16, 0x5678u16][..]);
        }

        // Four elements of four bytes each
        #[test]
        fn fourbyfour() {
            cfg_if! {
                if #[cfg(target_endian = "big")] {
                    let src = [0x00u8, 0xcc, 0x88, 0x44, 0xff, 0xbb, 0x77, 0x33,
                       0xee, 0xaa, 0x66, 0x22, 0xdd, 0x99, 0x55, 0x11];
                } else {
                    let src = [0x44, 0x88, 0xcc, 0x00, 0x33, 0x77, 0xbb, 0xff,
                       0x22, 0x66, 0xaa, 0xee, 0x11, 0x55, 0x99, 0xdd];
                }
            }
            let mut dst = [0u32; 4];
            unsafe { unshuffle(&src[..], &mut dst[..]) };
            assert_eq!(
                dst,
                &[0x11223344u32, 0x55667788, 0x99aabbcc, 0xddeeff00][..]
            );
        }

        /// unshuffle_bytes should be the inverse of shuffle_bytes
        #[rstest]
        fn inverse(
            #[values(2, 4, 8, 16, 18, 32, 36, 43, 47)] typesize: usize,
            #[values(64, 65, 256, 258, 1024, 1028, 4096, 4112)] len: usize,
        ) {
            let mut rng = rand::rng();

            let src = (0..len).map(|_| rng.random()).collect::<Vec<u8>>();
            let mut shuffled = vec![0u8; len];
            let mut dst = vec![0u8; len];
            unsafe {
                crate::generic::shuffle(typesize, len, src.as_ptr(), shuffled.as_mut_ptr());
                crate::generic::unshuffle(typesize, len, shuffled.as_ptr(), dst.as_mut_ptr());
            }
            assert_eq!(src, dst);
        }
    }
}
