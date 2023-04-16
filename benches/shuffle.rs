use std::{env, str::FromStr, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use shuffle::{shuffle_bytes, SimdImpl};

struct Spec {
    name:     &'static str,
    typesize: usize,
    size:     usize,
}
impl Spec {
    fn new(name: &'static str, typesize: usize, size: usize) -> Self {
        Spec {
            name,
            typesize,
            size,
        }
    }
}

fn select_impl() {
    match env::var("SHUFFLE_IMPL").map(|s| SimdImpl::from_str(s.as_str())) {
        Ok(Ok(impl_)) => {
            // Safe because we're single-threaded before main
            unsafe { shuffle::select_implementation(impl_) }
        }
        _ => (),
    }
}

fn shuffle_(c: &mut Criterion) {
    // Ideally select_impl would only be called directly from main.
    select_impl();

    let mut g = c.benchmark_group("shuffle");
    // These benchmarks are very small and I/O-lite.  Reduce Criterion's sampling time.
    g.warm_up_time(Duration::from_millis(100));
    g.measurement_time(Duration::from_millis(100));

    for spec in [
        Spec::new("two", 2, 4096),
        Spec::new("four", 4, 4096),
        Spec::new("16x256", 16, 256),
        Spec::new("sixteen", 16, 4096),
        Spec::new("fs-32", 32, 36844),
        Spec::new("fs-36", 36, 6564),
        Spec::new("ridt-43", 43, 17255),
        Spec::new("ridt-47", 47, 15240),
        Spec::new("alloct-18", 18, 11478),
    ]
    .iter()
    {
        g.throughput(Throughput::Bytes(spec.size as u64));
        let id = BenchmarkId::from_parameter(spec.name);
        g.bench_with_input(id, &spec, |b, spec| {
            let src = vec![0u8; spec.size];
            let mut dst = vec![0u8; spec.size];
            b.iter(|| shuffle_bytes(spec.typesize, &src[..], &mut dst[..]))
        });
    }
}

criterion_group!(benches, shuffle_);
criterion_main!(benches);
