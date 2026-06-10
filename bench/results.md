# Benchmark results

Generated from `bench/results.csv` by `tools/plot_bench.py`.

| section | method | N | threads | θ | ms/step | tree build ms | traverse ms |
|---|---|---:|---:|---:|---:|---:|---:|
| sweep | naive | 1,024 | 1 |  | 1.75 |  |  |
| sweep | naive | 4,096 | 1 |  | 28.22 |  |  |
| sweep | naive | 16,384 | 1 |  | 457.33 |  |  |
| sweep | naive_mt | 4,096 | 8 |  | 5.73 |  |  |
| sweep | naive_mt | 16,384 | 8 |  | 91.08 |  |  |
| sweep | naive_mt | 65,536 | 8 |  | 2421.68 |  |  |
| sweep | simd | 1,024 | 1 |  | 1.26 |  |  |
| sweep | simd | 4,096 | 1 |  | 16.45 |  |  |
| sweep | simd | 16,384 | 1 |  | 195.09 |  |  |
| sweep | simd | 65,536 | 1 |  | 3070.03 |  |  |
| sweep | simd_mt | 4,096 | 8 |  | 2.24 |  |  |
| sweep | simd_mt | 16,384 | 8 |  | 62.83 |  |  |
| sweep | simd_mt | 65,536 | 8 |  | 1579.59 |  |  |
| sweep | simd_mt | 131,072 | 8 |  | 5838.78 |  |  |
| sweep | bh_mt | 16,384 | 8 | 0.50 | 42.76 | 2.85 | 40.04 |
| sweep | bh_mt | 65,536 | 8 | 0.50 | 205.25 | 11.86 | 192.93 |
| sweep | bh_mt | 262,144 | 8 | 0.50 | 897.52 | 51.53 | 844.60 |
| sweep | bh_mt | 1,048,576 | 8 | 0.50 | 4039.17 | 185.92 | 3848.43 |
| scaling | simd_mt | 32,768 | 1 |  | 763.84 |  |  |
| scaling | simd_mt | 32,768 | 2 |  | 615.17 |  |  |
| scaling | simd_mt | 32,768 | 4 |  | 412.27 |  |  |
| scaling | simd_mt | 32,768 | 6 |  | 372.43 |  |  |
| scaling | simd_mt | 32,768 | 8 |  | 354.44 |  |  |
