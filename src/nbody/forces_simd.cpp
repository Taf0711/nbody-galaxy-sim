#include <arm_neon.h>

#include <cmath>

#include "nbody/forces.h"
#include "nbody/thread_pool.h"

namespace nbody {
namespace {

// 1/sqrt via NEON reciprocal-sqrt estimate plus two Newton-Raphson steps.
// vrsqrteq gives ~8 valid bits; each vrsqrtsq step roughly doubles that, so
// two steps recover near-full float precision while avoiding fsqrt + fdiv —
// the single biggest win over the scalar kernel.
inline float32x4_t rsqrt_nr2(float32x4_t v) {
  float32x4_t e = vrsqrteq_f32(v);
  e = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
  e = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
  return e;
}

void simd_range(Bodies& b, float eps2, std::size_t i0, std::size_t i1) {
  const std::size_t n = b.size();
  const float* x = b.x.data();
  const float* y = b.y.data();
  const float* z = b.z.data();
  const float* m = b.m.data();
  const float32x4_t veps2 = vdupq_n_f32(eps2);

  for (std::size_t i = i0; i < i1; ++i) {
    const float32x4_t xi = vdupq_n_f32(x[i]);
    const float32x4_t yi = vdupq_n_f32(y[i]);
    const float32x4_t zi = vdupq_n_f32(z[i]);

    // Two independent accumulator sets: the rsqrt chain is long, so running
    // two j-blocks in flight keeps the M-series pipelines fed.
    float32x4_t ax0 = vdupq_n_f32(0.0f), ay0 = ax0, az0 = ax0;
    float32x4_t ax1 = ax0, ay1 = ax0, az1 = ax0;

    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
      const float32x4_t dxa = vsubq_f32(vld1q_f32(x + j), xi);
      const float32x4_t dya = vsubq_f32(vld1q_f32(y + j), yi);
      const float32x4_t dza = vsubq_f32(vld1q_f32(z + j), zi);
      const float32x4_t dxb = vsubq_f32(vld1q_f32(x + j + 4), xi);
      const float32x4_t dyb = vsubq_f32(vld1q_f32(y + j + 4), yi);
      const float32x4_t dzb = vsubq_f32(vld1q_f32(z + j + 4), zi);

      const float32x4_t r2a =
          vfmaq_f32(vfmaq_f32(vfmaq_f32(veps2, dxa, dxa), dya, dya), dza, dza);
      const float32x4_t r2b =
          vfmaq_f32(vfmaq_f32(vfmaq_f32(veps2, dxb, dxb), dyb, dyb), dzb, dzb);

      const float32x4_t inva = rsqrt_nr2(r2a);
      const float32x4_t invb = rsqrt_nr2(r2b);
      const float32x4_t sa =
          vmulq_f32(vld1q_f32(m + j), vmulq_f32(vmulq_f32(inva, inva), inva));
      const float32x4_t sb = vmulq_f32(
          vld1q_f32(m + j + 4), vmulq_f32(vmulq_f32(invb, invb), invb));

      ax0 = vfmaq_f32(ax0, sa, dxa);
      ay0 = vfmaq_f32(ay0, sa, dya);
      az0 = vfmaq_f32(az0, sa, dza);
      ax1 = vfmaq_f32(ax1, sb, dxb);
      ay1 = vfmaq_f32(ay1, sb, dyb);
      az1 = vfmaq_f32(az1, sb, dzb);
    }
    for (; j + 4 <= n; j += 4) {
      const float32x4_t dx = vsubq_f32(vld1q_f32(x + j), xi);
      const float32x4_t dy = vsubq_f32(vld1q_f32(y + j), yi);
      const float32x4_t dz = vsubq_f32(vld1q_f32(z + j), zi);
      const float32x4_t r2 =
          vfmaq_f32(vfmaq_f32(vfmaq_f32(veps2, dx, dx), dy, dy), dz, dz);
      const float32x4_t inv = rsqrt_nr2(r2);
      const float32x4_t s =
          vmulq_f32(vld1q_f32(m + j), vmulq_f32(vmulq_f32(inv, inv), inv));
      ax0 = vfmaq_f32(ax0, s, dx);
      ay0 = vfmaq_f32(ay0, s, dy);
      az0 = vfmaq_f32(az0, s, dz);
    }

    float ax = vaddvq_f32(vaddq_f32(ax0, ax1));
    float ay = vaddvq_f32(vaddq_f32(ay0, ay1));
    float az = vaddvq_f32(vaddq_f32(az0, az1));

    for (; j < n; ++j) {  // scalar tail (n not a multiple of 4)
      const float dx = x[j] - x[i];
      const float dy = y[j] - y[i];
      const float dz = z[j] - z[i];
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

void forces_simd(Bodies& b, float eps2, ThreadPool* pool) {
  if (pool) {
    pool->parallel_for(0, b.size(), 256, [&](std::size_t lo, std::size_t hi) {
      simd_range(b, eps2, lo, hi);
    });
  } else {
    simd_range(b, eps2, 0, b.size());
  }
}

}  // namespace nbody
