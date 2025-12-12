# byteshuffle

SIMD-accelerated byte shuffle/unshuffle routines

The byte-shuffle is a very efficient way to improve the compressibility of data
that consists of an array of fixed-size objects.  It rearranges the array in
order to group all elements' least significant bytes together, most-significant
bytes together, and everything in between.  Since real applications' arrays
often contain consecutive elements that are closely correlated with each other,
this filter frequently results in lengthy continuous runs of identical bytes.
Such runs are highly compressible by general-purpose compression libraries like
gzip, lz4, etc.

![Build Status](https://api.cirrus-ci.com/github/asomers/byteshuffle.svg)
[![Crates.io](https://img.shields.io/crates/v/byteshuffle.svg)](https://crates.io/crates/byteshuffle)

[Documentation](https://docs.rs/crate/byteshuffle)

# Platforms

This crate is OS-agnostic.  But it relies on compiler intrinsics for high
performance.  So while it will run anywhere, it will only achieve good
performance on x86_64 and x86 processors.  Implementations for additional
architectures are welcome.

# SIMD implementations

SIMD acceleration is implemented for certain type sizes and certain instruction
sets only.  Currently accelerated operations are:

| Typesize | SSE2 | AVX2 | AVX512F      |
| -------- | ---- | ---- | ------------ |       
|       2B | both | both | shuffle only |
|      16B | both | both | shuffle only |
|    > 16B | both | both |              |
|     x 4B |      |      | shuffle only |

# Minimum Supported Rust Version (MSRV)

Byteshuffle is supported on Rust 1.89.0 and higher.  Byteshuffle's MSRV will
not be changed in the future without bumping the major or minor version.

# License

`byteshuffle` is primarily distributed under the terms of the BSD 3-clause license.

See LICENSE for details.

# Acknowledgements

The [blosc](https://www.blosc.org) project was the original inspiration for this library.
Blosc is a C library intended primarily for HPC users, and it implements a
shuffle filter, among many other things.  This crate is a reimplementation of
Blosc's shuffle filter.  Some of the SIMD-optimized functions were directly
translated from C-Blosc's C source, and others were written fresh.  But it
excludes every other feature of Blosc.
