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
//! let shuffled = shuffle(2, &IN);
//! assert_ne!(IN, &shuffled[..]);
//! // In normal use, you would now serialize `shuffled`.  Then compress it, and later decompress
//! // and deserialize it.  Then unshuffle like below.
//! let unshuffled = unshuffle(2, &shuffled);
//! assert_eq!(IN, &unshuffled[..]);
//! ```
//!
//! # Crate Features
//!
//! This crate has a "nightly" feature.  It enables methods that require the use of types from the
//! standard library that aren't yet stabilized.
#![cfg_attr(feature = "nightly", feature(core_io_borrowed_buf))]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "nightly")]
use std::io::BorrowedCursor;
use std::{mem, str::FromStr};

use cfg_if::cfg_if;
use ctor::ctor;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512f;
mod generic;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2;

type LlFunc = unsafe fn(usize, usize, *const u8, *mut u8);
static mut IMPL: (LlFunc, LlFunc) = (generic::shuffle, generic::unshuffle);

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
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            match impl_ {
                SimdImpl::Auto => {
                    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"){
                        unsafe { IMPL = (avx512f::shuffle, avx2::unshuffle); }
                    } else if is_x86_feature_detected!("avx2") {
                        unsafe { IMPL = (avx2::shuffle, avx2::unshuffle); }
                    } else if is_x86_feature_detected!("sse2") {
                        unsafe { IMPL = (sse2::shuffle, sse2::unshuffle); }
                    } else {
                        unsafe { IMPL = (generic::shuffle, generic::unshuffle); }
                    }
                },
                SimdImpl::Generic => unsafe { IMPL = (generic::shuffle, generic::unshuffle); },
                SimdImpl::Sse2 => unsafe { IMPL = (sse2::shuffle, sse2::unshuffle); },
                SimdImpl::Avx2 => unsafe { IMPL = (avx2::shuffle, sse2::unshuffle); },
                SimdImpl::Avx512F => unsafe { IMPL = (avx512f::shuffle, sse2::unshuffle); },
            }
        } else {
            let _ = impl_;
            unsafe { IMPL = (generic::shuffle, generic::unshuffle); }
        }
    }
}

/// Shuffle an array of fixed-size objects.
///
/// The result will be a shuffled byte array.  Be aware that this byte array will usually not be
/// portable.  It can only be unshuffled by the same computer architecture as the one that shuffled
/// it, and sometimes only by the same compiler version.  If portability is desired, then first
/// serialize the data and then use [`shuffle`].
///
/// # Examples
/// ```
/// # use byteshuffle::*;
/// const IN: [u16; 4] = [0x01, 0x02, 0x03, 0x04];
/// let out = shuffle_objects(&IN);
#[cfg_attr(
    target_endian = "little",
    doc = "assert_eq!(out, [0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00]);"
)]
#[cfg_attr(
    target_endian = "big",
    doc = "assert_eq!(out, [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04]);"
)]
/// ```
pub fn shuffle_objects<T: Copy>(src: &[T]) -> Vec<u8> {
    let ts = mem::size_of::<T>();
    assert!(ts > 1, "No point shuffling plain [u8]");
    let mut dst = Vec::with_capacity(mem::size_of_val(src));
    // Safe because we src and dst have same length in bytes
    unsafe {
        IMPL.0(
            ts,
            mem::size_of_val(src),
            src.as_ptr() as *const u8,
            dst.as_mut_ptr(),
        );
        dst.set_len(mem::size_of_val(src));
    };
    dst
}

/// Shuffle a byte array whose contents are known to approximately repeat with
/// period `typesize`.
///
/// # Examples
/// ```
/// # use byteshuffle::*;
/// const IN: [u8; 8] = [0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04];
/// let out = shuffle(2, &IN);
/// assert_eq!(out, [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04]);
/// ```
pub fn shuffle(typesize: usize, src: &[u8]) -> Vec<u8> {
    let mut dst = Vec::with_capacity(src.len());
    // Safe because src and dst have the same capacity
    unsafe {
        IMPL.0(typesize, src.len(), src.as_ptr(), dst.as_mut_ptr());
        dst.set_len(src.len());
    }
    assert_eq!(src.len(), dst.len());
    dst
}

/// Like [`shuffle`], but allows the caller to control allocation for the output.
///
/// # Examples
/// ```
/// # use byteshuffle::*;
/// const IN: [u8; 8] = [0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04];
/// let mut out = [0u8; 8];
/// shuffle_into(2, &IN, &mut out);
/// assert_eq!(out, [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04]);
/// ```
pub fn shuffle_into(typesize: usize, src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    // Safe because src and dst have the same length
    unsafe {
        IMPL.0(typesize, src.len(), src.as_ptr(), dst.as_mut_ptr());
    }
}

/// Like [`shuffle_into`], but works with uninitialized output buffers.
///
/// # Example
/// ```
/// #![cfg_attr(feature = "nightly", feature(core_io_borrowed_buf))]
/// use byteshuffle::*;
/// use std::io::BorrowedBuf;
/// use std::mem::MaybeUninit;
///
/// const IN: [u8; 8] = [0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04];
/// let mut outvec = [MaybeUninit::uninit(); 8];
/// let mut buf = BorrowedBuf::from(&mut outvec[..]);
/// shuffle_buf(2, &IN, buf.unfilled());
/// assert_eq!(buf.filled(), [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04]);
/// ```
#[cfg(feature = "nightly")]
#[cfg_attr(docsrs, doc(cfg(feature = "nightly")))]
pub fn shuffle_buf(typesize: usize, src: &[u8], mut buf: BorrowedCursor<'_>) {
    assert!(buf.capacity() >= src.len());
    unsafe {
        let dst: *mut u8 = buf.as_mut().as_mut_ptr().cast();
        IMPL.0(typesize, src.len(), src.as_ptr(), dst);
        buf.advance(src.len());
    }
}

/// Unshuffle an array of fixed-size objects.
///
/// # Safety
///
/// The `src` must've originally been produced by [`shuffle`], by a program that uses the exact
/// same byte layout as this one.  That means the same wordsize, same endianness, and may even
/// require the same compiler version.  If you can't guarantee those conditions, then serialize and
/// use [`shuffle`]/[`unshuffle`] instead.
///
/// # Examples
/// ```
/// # use byteshuffle::*;
/// const IN: [u16; 4] = [0x01, 0x02, 0x03, 0x04];
/// let mut shuffled = shuffle_objects(&IN);
/// let out = unsafe { unshuffle_objects(&shuffled) };
/// assert_eq!(IN, &out[..]);
/// ```
pub unsafe fn unshuffle_objects<T: Copy>(src: &[u8]) -> Vec<T> {
    let ts = mem::size_of::<T>();
    assert!(ts > 1, "No point shuffling plain [u8]");
    let mut dst = Vec::with_capacity(src.len() / ts);
    // Safe because we src and dst have same length in bytes
    unsafe {
        IMPL.1(
            ts,
            mem::size_of_val(src),
            src.as_ptr(),
            dst.as_mut_ptr() as *mut u8,
        );
        dst.set_len(src.len() / ts);
    }
    assert_eq!(mem::size_of_val(src), mem::size_of_val(&dst[..]));
    dst
}

/// Unshuffle a byte array whose contents are known to approximately repeat with
/// period `typesize`.
///
/// # Examples
/// ```
/// # use byteshuffle::*;
/// const IN: [u8; 8] = [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04];
/// let out = unshuffle(2, &IN);
/// assert_eq!(out, [0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04]);
/// ```
pub fn unshuffle(typesize: usize, src: &[u8]) -> Vec<u8> {
    let mut dst = Vec::with_capacity(src.len());
    unsafe {
        IMPL.1(typesize, src.len(), src.as_ptr(), dst.as_mut_ptr());
        dst.set_len(src.len());
    }
    assert_eq!(src.len(), dst.len());
    dst
}

/// Like [`unshuffle`], but allows the caller to control allocation for the destination.
///
/// # Example
/// ```
/// use byteshuffle::*;
///
/// const IN: [u8; 8] = [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04];
/// let mut out = [0u8; 8];
/// unshuffle_into(2, &IN, &mut out);
/// assert_eq!(out, [0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04]);
/// ```
pub fn unshuffle_into(typesize: usize, src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    unsafe {
        IMPL.1(typesize, src.len(), src.as_ptr(), dst.as_mut_ptr());
    }
}

/// Like [`unshuffle_into`], but works with uninitialized output buffers.
///
/// # Example
/// ```
/// #![cfg_attr(feature = "nightly", feature(core_io_borrowed_buf))]
/// use byteshuffle::*;
/// use std::io::BorrowedBuf;
/// use std::mem::MaybeUninit;
///
/// const IN: [u8; 8] = [0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04];
/// let mut outvec = [MaybeUninit::uninit(); 8];
/// let mut buf = BorrowedBuf::from(&mut outvec[..]);
/// unshuffle_buf(2, &IN, buf.unfilled());
/// assert_eq!(buf.filled(), [0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04]);
/// ```
#[cfg(feature = "nightly")]
#[cfg_attr(docsrs, doc(cfg(feature = "nightly")))]
pub fn unshuffle_buf(typesize: usize, src: &[u8], mut buf: BorrowedCursor<'_>) {
    assert!(buf.capacity() >= src.len());
    unsafe {
        let dst: *mut u8 = buf.as_mut().as_mut_ptr().cast();
        IMPL.1(typesize, src.len(), src.as_ptr(), dst);
        buf.advance(src.len());
    }
}

#[cfg(test)]
mod t {
    use super::*;

    /// Test the type-based shuffle_objects API
    mod shuffle_objects {
        use super::*;

        // Two elements of two bytes each
        #[test]
        fn twobytwo() {
            let src = [0x1234u16, 0x5678u16];
            let dst = shuffle_objects(&src[..]);
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
            let dst = shuffle_objects(&src[..]);
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

    mod shuffle {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        use rand::Rng;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        use rstest::rstest;

        /// Compare optimized results against generic results
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[rstest]
        #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"),
            case::sse2(crate::sse2::shuffle, is_x86_feature_detected!("sse2")))]
        #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"),
            case::avx2(crate::avx2::shuffle, is_x86_feature_detected!("avx2")))]
        #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"),
            case::avx512f(crate::avx512f::shuffle,
                        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
            )
        )]
        fn compare(
            #[values(2, 4, 8, 13, 16, 18, 32, 36, 43, 47)] typesize: usize,
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
    mod unshuffle_objects {
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
            let dst = unsafe { unshuffle_objects::<u16>(&src[..]) };
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
            let dst = unsafe { unshuffle_objects::<u32>(&src[..]) };
            assert_eq!(
                dst,
                &[0x11223344u32, 0x55667788, 0x99aabbcc, 0xddeeff00][..]
            );
        }
    }

    mod unshuffle {
        use rand::Rng;
        use rstest::rstest;

        /// Compare optimized results against generic results
        #[rstest]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"),
            case::sse2(crate::sse2::unshuffle, is_x86_feature_detected!("sse2")))]
        #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"),
            case::avx2(crate::avx2::unshuffle, is_x86_feature_detected!("avx2")))]
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
                crate::generic::unshuffle(typesize, len, src.as_ptr(), generic_dst.as_mut_ptr());
                f(typesize, len, src.as_ptr(), opt_dst.as_mut_ptr());
            }
            assert_eq!(generic_dst, opt_dst);
        }

        /// unshuffle should be the inverse of shuffle
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
