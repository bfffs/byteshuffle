//! Generic non-optimized stuff

use std::ptr;

/// Generic non-optimized shuffle routine
///
/// # Safety
/// src and dst must both be of exactly len bytes long.
pub unsafe fn shuffle(
    typesize: usize,
    len: usize,
    src: *const u8,
    dst: *mut u8)
{
    shuffle_partial(typesize, 0, len, src, dst)
}

/// Shuffle the tail end of a mostly-shuffled block of data.  Begin `start` into
/// the buffer.
pub unsafe fn shuffle_partial(
    typesize: usize,
    start: usize,
    len: usize,
    src: *const u8,
    dst: *mut u8)
{
    let vectorizable_elements = start / typesize;
    let quot = len / typesize;
    let rem = len % typesize;

    for j in 0..typesize {
        for i in vectorizable_elements..quot {
            *dst.add(j * quot + i) = *src.add(i * typesize + j)
        }
    }
    ptr::copy_nonoverlapping(src.add(len - rem), dst.add(len - rem), rem);
}

/// Generic non-optimized unshuffle routine
///
/// # Safety
/// src and dst must both be of exactly len bytes long.
pub unsafe fn unshuffle(
    typesize: usize,
    len: usize,
    src: *const u8,
    dst: *mut u8)
{
    let quot = len / typesize;
    let rem = len % typesize;

    for i in 0..quot {
        for j in 0..typesize {
            *dst.add(i * typesize + j) = *src.add(j * quot + i)
        }
    }
    ptr::copy_nonoverlapping(src.add(len - rem), dst.add(len - rem), rem);
}

#[cfg(test)]
mod t {
    use super::*;
    
    mod shuffle {
        use super::*;

        // Two elements of two bytes each plus remainder
        #[test]
        fn twobytwoplusone() {
            let src = [0x34u8, 0x12, 0x78, 0x56, 0x9a];
            let mut dst = [0u8; 5];
            unsafe{ shuffle(2, src.len(), src.as_ptr(), dst.as_mut_ptr()) };
            assert_eq!(dst, &[0x34, 0x78, 0x12, 0x56, 0x9a][..]);
        }
    }

    mod unshuffle {
        use super::*;

        // Two elements of two bytes each plus remainder
        #[test]
        fn twobytwoplusone() {
            let src = [0x34, 0x78, 0x12, 0x56, 0x9a];
            let mut dst = [0u8; 5];
            unsafe{ unshuffle(2, src.len(), src.as_ptr(), dst.as_mut_ptr())};
            assert_eq!(dst, &[0x34u8, 0x12, 0x78, 0x56, 0x9a][..]);
        }
    }
}
