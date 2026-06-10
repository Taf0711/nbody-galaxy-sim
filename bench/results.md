# Benchmark results

Generated from `bench/results.csv` by `tools/plot_bench.py`.

| section | method | N | threads | θ | ms/step | tree build ms | traverse ms |
|---|---|---:|---:|---:|---:|---:|---:|
| sweep | naive | 1,024 | 1 |  | 1.63 |  |  |
| sweep | naive | 4,096 | 1 |  | 16.89 |  |  |
| sweep | naive | 16,384 | 1 |  | 275.96 |  |  |
| sweep | naive_mt | 4,096 | 8 |  | 4.41 |  |  |
| sweep | naive_mt | 16,384 | 8 |  | 71.95 |  |  |
| sweep | naive_mt | 65,536 | 8 |  | 1016.94 |  |  |
| sweep | simd | 1,024 | 1 |  | 0.46 |  |  |
| sweep | simd | 4,096 | 1 |  | 7.41 |  |  |
| sweep | simd | 16,384 | 1 |  | 120.25 |  |  |
| sweep | simd | 65,536 | 1 |  | 1861.01 |  |  |
| sweep | simd_mt | 4,096 | 8 |  | 1.87 |  |  |
| sweep | simd_mt | 16,384 | 8 |  | 31.97 |  |  |
| sweep | simd_mt | 65,536 | 8 |  | 475.51 |  |  |
| sweep | simd_mt | 131,072 | 8 |  | 2043.57 |  |  |
| sweep | bh_mt | 16,384 | 8 | 0.50 | 15.96 | 1.10 | 14.79 |
| sweep | bh_mt | 65,536 | 8 | 0.50 | 81.94 | 5.09 | 76.90 |
| sweep | bh_mt | 262,144 | 8 | 0.50 | 375.78 | 22.00 | 354.06 |
| sweep | bh_mt | 1,048,576 | 8 | 0.50 | 1657.71 | 95.29 | 1558.78 |
| sweep | bh_quad_mt | 16,384 | 8 | 0.65 | 15.11 | 1.20 | 13.89 |
| sweep | bh_quad_mt | 65,536 | 8 | 0.65 | 78.06 | 5.34 | 72.42 |
| sweep | bh_quad_mt | 262,144 | 8 | 0.65 | 345.79 | 22.60 | 322.33 |
| sweep | bh_quad_mt | 1,048,576 | 8 | 0.65 | 1359.84 | 99.29 | 1261.80 |
| scaling | simd_mt | 32,768 | 1 |  | 457.45 |  |  |
| scaling | simd_mt | 32,768 | 2 |  | 247.42 |  |  |
| scaling | simd_mt | 32,768 | 4 |  | 144.39 |  |  |
| scaling | simd_mt | 32,768 | 6 |  | 135.48 |  |  |
| scaling | simd_mt | 32,768 | 8 |  | 126.99 |  |  |
