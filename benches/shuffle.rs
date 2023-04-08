use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use shuffle::{shuffle, shuffle_bytes};


struct Spec {
    name: &'static str,
    typesize: usize,
    size: usize
}
impl Spec {
    fn new(name: &'static str, typesize: usize, size: usize) -> Self {
        Spec{name, typesize, size}
    }
}

fn shuffle_(c: &mut Criterion) {
    let mut g = c.benchmark_group("shuffle");

    for spec in [
        Spec::new("two", 2, 4096),
        Spec::new("four", 4, 4096),
        Spec::new("fs-32", 32, 36844),
        Spec::new("fs-36", 36, 6564),
        Spec::new("ridt-43", 43, 17255),
        Spec::new("ridt-47", 47, 15240),
        Spec::new("alloct-18", 18, 11478)].iter()
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
