# Benchmark results

Generated from `bench/results.csv` by `tools/plot_bench.py`.

| section | method | N | threads | θ | ms/step | tree build ms | traverse ms |
|---|---|---:|---:|---:|---:|---:|---:|
| sweep | naive | 1,024 | 1 |  | 0.99 |  |  |
| sweep | naive | 4,096 | 1 |  | 16.57 |  |  |
| sweep | naive | 16,384 | 1 |  | 273.59 |  |  |
| sweep | naive_mt | 4,096 | 8 |  | 4.09 |  |  |
| sweep | naive_mt | 16,384 | 8 |  | 64.06 |  |  |
| sweep | naive_mt | 65,536 | 8 |  | 1081.78 |  |  |
| sweep | simd | 1,024 | 1 |  | 0.52 |  |  |
| sweep | simd | 4,096 | 1 |  | 7.03 |  |  |
| sweep | simd | 16,384 | 1 |  | 114.69 |  |  |
| sweep | simd | 65,536 | 1 |  | 1807.07 |  |  |
| sweep | simd_mt | 4,096 | 8 |  | 1.84 |  |  |
| sweep | simd_mt | 16,384 | 8 |  | 27.16 |  |  |
| sweep | simd_mt | 65,536 | 8 |  | 433.33 |  |  |
| sweep | simd_mt | 131,072 | 8 |  | 1846.48 |  |  |
| sweep | bh_mt | 16,384 | 8 | 0.50 | 14.07 | 1.01 | 13.00 |
| sweep | bh_mt | 65,536 | 8 | 0.50 | 69.78 | 4.60 | 65.29 |
| sweep | bh_mt | 262,144 | 8 | 0.50 | 306.54 | 19.75 | 285.39 |
| sweep | bh_mt | 1,048,576 | 8 | 0.50 | 1479.47 | 94.53 | 1382.00 |
| scaling | simd_mt | 32,768 | 1 |  | 461.21 |  |  |
| scaling | simd_mt | 32,768 | 2 |  | 233.54 |  |  |
| scaling | simd_mt | 32,768 | 4 |  | 134.50 |  |  |
| scaling | simd_mt | 32,768 | 6 |  | 131.17 |  |  |
| scaling | simd_mt | 32,768 | 8 |  | 126.91 |  |  |
