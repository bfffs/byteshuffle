use std::{mem, ptr};


/// Generic non-optimized shuffle routine
///
/// # Safety
/// src and dst must both be of exactly len bytes long.
unsafe fn shuffle_generic(
    typesize: usize,
    len: usize,
    src: *const u8,
    dst: *mut u8)
{
    let quot = len / typesize;
    let rem = len % typesize;

    for j in 0..typesize {
        for i in 0..quot {
            *dst.add(j * quot + i) = *src.add(i * typesize + j)
        }
    }
    ptr::copy_nonoverlapping(src.add(len - rem), dst.add(len - rem), rem);
}

/// Shuffle an array of fixed-size objects
pub fn shuffle<T: Copy>(src: &[T], dst: &mut [T]) {
    assert_eq!(src.len(), dst.len());
    let ts = mem::size_of::<T>();
    assert!(ts > 1, "No point shuffling plain [u8]");
    // Safe because of the first assertion.
    unsafe {
        shuffle_generic(ts, src.len() * ts,
            src.as_ptr() as *const u8, dst.as_ptr() as *mut u8);
    }
}

/// Shuffle a byte array whose contents are known to approximately repeat with
/// period `typesize`.
pub fn shuffle_bytes(typesize: usize, src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    // Safe because of the first assertion.
    unsafe {
        shuffle_generic(typesize, src.len(),
            src.as_ptr() as *const u8, dst.as_ptr() as *mut u8);
    }
}

#[cfg(test)]
mod t {
    use super::*;
    
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
        use super::*;

        // Two elements of two bytes each plus remainder
        #[test]
        fn twobytwoplusone() {
            let src = [0x34u8, 0x12, 0x78, 0x56, 0x9a];
            let mut dst = [0u8; 5];
            shuffle_bytes(2, &src[..], &mut dst[..]);
            assert_eq!(dst, &[0x34, 0x78, 0x12, 0x56, 0x9a][..]);
        }
    }
}
