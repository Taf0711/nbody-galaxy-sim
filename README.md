# N-Body Galaxy Simulator

Real-time gravitational simulation, twice: a CPU engine in C++ pushed from a
naive O(n²) loop to a multithreaded, NEON-vectorized Barnes-Hut tree that
steps **a million bodies in ~1.5 s**, and a WebGPU compute demo that runs
tens of thousands of bodies at 60 fps **entirely in your browser** — fling
mass into a galaxy collision and watch it evolve.

**[▶ Live demo](https://taf0711.github.io/nbody-galaxy-sim/)** — needs WebGPU
(Chrome, Edge, or Safari 18+). Drag to orbit, scroll to zoom,
**shift-drag to fling mass into the simulation**.

![galaxy collision in the WebGPU demo](assets/screenshot.png)

## The optimization story

Same physics at every step — softened gravity, kick-drift-kick leapfrog —
measured at n = 16,384 on an Apple M2 (4P+4E cores, clang -O3, no fast-math):

| stage | ms/step | speedup |
|---|---:|---:|
| naive all-pairs, scalar, 1 thread | 273.6 | 1.0× |
| + thread pool (8 threads) | 64.1 | 4.3× |
| NEON SIMD kernel, 1 thread | 114.7 | 2.4× |
| SIMD + threads | 27.2 | 10.1× |
| **Barnes-Hut (θ = 0.5) + threads** | **14.1** | **19.4×** |

![ms/step vs N, all methods](assets/bench_sweep.svg)

The brute-force curves grow as n²; Barnes-Hut grows as n·log n. The gap is
already 6× at 65k bodies and grows without bound:

| N | Barnes-Hut | all-pairs SIMD+threads |
|---:|---:|---:|
| 65,536 | 69.8 ms | 433 ms |
| 262,144 | 307 ms | ~7,400 ms (extrapolated) |
| 1,048,576 | 1,479 ms | ~118,000 ms (extrapolated) |

At a million bodies the tree build (Morton sort + construction) is only 95 ms
of the 1.48 s step — traversal dominates, which is exactly where you want the
time going.

![thread scaling](assets/bench_scaling.svg)

Two honest footnotes that took debugging to learn:

* The "naive scalar" baseline is not as naive as it looks — clang
  auto-vectorizes it to ~1 G pair-interactions/s. The hand-written kernel's
  2.4× on top comes from what the compiler *won't* do without
  `-ffast-math`: replacing `sqrt + divide` with NEON's reciprocal-sqrt
  estimate plus two Newton-Raphson steps, and running four independent
  accumulator streams so the latency chains overlap.
* Thread scaling tops out at ~3.6×, almost all of it from the four P-cores.
  And on macOS, two things will silently wreck a parallel benchmark:
  worker threads without an explicit QoS class can get parked on E-cores,
  and Low Power Mode caps the whole pool at ~2× while single-thread numbers
  look normal. (Full data: [bench/results.md](bench/results.md).)

## How it works

**Physics.** Plummer-softened gravity (G = 1), kick-drift-kick leapfrog.
Leapfrog is symplectic — energy errors stay bounded instead of accumulating —
and softening makes the i = j self-interaction term exactly zero, so every
inner loop is branch-free: no `if (i != j)`, no special cases, just FMAs.

**SIMD kernel** (`src/nbody/forces_simd.cpp`). Bodies live in
structure-of-arrays layout so `vld1q_f32` grabs four bodies per instruction.
The core trick:

```c++
// ~8 valid bits from the estimate; each Newton-Raphson step doubles that.
inline float32x4_t rsqrt_nr2(float32x4_t v) {
  float32x4_t e = vrsqrteq_f32(v);
  e = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
  e = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
  return e;
}
```

That chain is ~7 dependent ops, so one stream leaves the M2's four NEON pipes
mostly idle — the kernel processes four j-blocks in flight with independent
accumulators, which is worth as much as the vectorization itself.

**Thread pool** (`src/nbody/thread_pool.h`). Chunked dynamic parallel-for:
threads pull ranges off a shared atomic cursor, so uneven work (tree
traversal depth varies per body) balances itself. The calling thread
participates, and workers pin their QoS class so the macOS scheduler keeps
them on performance cores.

**Barnes-Hut** (`src/nbody/barnes_hut.cpp`). Not a pointer octree: each step,
bodies are sorted by 63-bit Morton code and the tree is built by recursively
splitting code ranges on 3-bit digits. Every node's bodies are then a
*contiguous range* of the sorted arrays, which buys three things:

1. leaf interactions run through the same NEON kernel shape as brute force,
2. children sit after parents in one flat array, so centers of mass are
   computed in a single reverse sweep with no recursion,
3. bodies in Morton order make near-identical tree walks, so the parallel
   traversal stays cache-friendly.

A body never approximates a node it is inside (that would add a spurious
self-force through the node's center of mass) — and setting θ = 0 forces the
tree to degenerate into exact all-pairs, which is the test that catches any
lost or double-counted body in the partition logic.

| θ | RMS force error vs exact | ms/step @ 16k |
|---:|---:|---:|
| 0.5 | 3.3 × 10⁻³ | 14.1 |
| 0.75 | 1.1 × 10⁻² | faster still |

**WebGPU demo** (`web/`). The same KDK scheme in WGSL. The force pass is the
classic tiled all-pairs kernel: 256-thread workgroups stage 256-body tiles
through workgroup memory, so each global position is read once per workgroup
instead of once per thread — the M2's GPU sustains ~64 billion
pair-interactions/s, ~6× the full 8-thread CPU SIMD rate. Rendering is
additive gaussian sprites (colored by speed) into an rgba16float target with
an exponential tonemap, so dense regions bloom naturally. No build step, no
dependencies — three ES modules and two shaders.

Why brute force on the GPU instead of porting Barnes-Hut? Divergent tree
traversal is a poor fit for lockstep GPU execution, and at the body counts a
web page wants (≤131k), tiled brute force already saturates the ALUs while
staying simple enough to verify. The algorithmic story lives in the CPU
phases; the GPU's job is to be fast and beautiful.

## Correctness

`make test` — every optimized path is validated against the naive reference,
and the physics against analytic behavior:

* two-body circular orbit: radius drift < 2 × 10⁻⁵ over 5 orbits
* Plummer sphere energy conserved to |ΔE/E| ~ 10⁻⁷ over 200 steps,
  momentum to ~10⁻⁹
* SIMD vs naive: max relative error 2 × 10⁻⁶
* threaded vs serial: bitwise identical
* Barnes-Hut θ = 0 vs naive: exact (tree partition correctness)
* Barnes-Hut θ = 0.7 energy conserved to ~10⁻⁵ over 100 steps

## Build & run

```sh
make test                        # build + physics test suite (~10 s)
make bench                       # full benchmark sweep -> bench/results.csv
python3 tools/plot_bench.py      # regenerate charts + bench/results.md
cd web && python3 -m http.server # demo at http://localhost:8000
```

Requires clang++ on Apple Silicon (NEON intrinsics) and any Python 3 for the
charts. The web demo is static files — host it anywhere.

## Hardware

All numbers: Apple M2 (MacBook Air, 4 performance + 4 efficiency cores),
Apple clang 17, `-O3` without `-ffast-math`, macOS Low Power Mode off, on a
cool idle machine — sustained all-core NEON heats a fanless laptop enough to
throttle within minutes, which is itself a benchmarking lesson.
