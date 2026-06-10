#include <cmath>

#include "nbody/forces.h"
#include "nbody/thread_pool.h"

namespace nbody {
namespace {

// Phase 1 baseline: scalar all-pairs. Deliberately straightforward — this is
// the reference both for correctness (tests compare every other kernel
// against it) and for the benchmark story.
void naive_range(Bodies& b, float eps2, std::size_t i0, std::size_t i1) {
  const std::size_t n = b.size();
  const float* x = b.x.data();
  const float* y = b.y.data();
  const float* z = b.z.data();
  const float* m = b.m.data();
  for (std::size_t i = i0; i < i1; ++i) {
    const float xi = x[i], yi = y[i], zi = z[i];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
      const float dx = x[j] - xi;
      const float dy = y[j] - yi;
      const float dz = z[j] - zi;
      const float r2 = dx * dx + dy * dy + dz * dz + eps2;
      const float inv = 1.0f / std::sqrt(r2);
      const float s = m[j] * inv * inv * inv;
      ax += s * dx;
      ay += s * dy;
      az += s * dz;
    }
    b.ax[i] = ax;
    b.ay[i] = ay;
    b.az[i] = az;
  }
}

}  // namespace

void forces_naive(Bodies& b, float eps2, ThreadPool* pool) {
  if (pool) {
    pool->parallel_for(0, b.size(), 256, [&](std::size_t lo, std::size_t hi) {
      naive_range(b, eps2, lo, hi);
    });
  } else {
    naive_range(b, eps2, 0, b.size());
  }
}

}  // namespace nbody
