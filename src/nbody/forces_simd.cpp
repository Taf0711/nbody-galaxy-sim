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

    // Four independent accumulator sets. The rsqrt+2NR chain is ~7 dependent
    // ops; a single stream leaves the M-series' four NEON pipes mostly idle
    // (measured ~1.4 Gpair/s with two streams vs ~0.6 scalar). Four in-flight
    // j-blocks overlap the latency chains.
    float32x4_t ax0 = vdupq_n_f32(0.0f), ay0 = ax0, az0 = ax0;
    float32x4_t ax1 = ax0, ay1 = ax0, az1 = ax0;
    float32x4_t ax2 = ax0, ay2 = ax0, az2 = ax0;
    float32x4_t ax3 = ax0, ay3 = ax0, az3 = ax0;

    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) {
      const float32x4_t dxa = vsubq_f32(vld1q_f32(x + j), xi);
      const float32x4_t dya = vsubq_f32(vld1q_f32(y + j), yi);
      const float32x4_t dza = vsubq_f32(vld1q_f32(z + j), zi);
      const float32x4_t dxb = vsubq_f32(vld1q_f32(x + j + 4), xi);
      const float32x4_t dyb = vsubq_f32(vld1q_f32(y + j + 4), yi);
      const float32x4_t dzb = vsubq_f32(vld1q_f32(z + j + 4), zi);
      const float32x4_t dxc = vsubq_f32(vld1q_f32(x + j + 8), xi);
      const float32x4_t dyc = vsubq_f32(vld1q_f32(y + j + 8), yi);
      const float32x4_t dzc = vsubq_f32(vld1q_f32(z + j + 8), zi);
      const float32x4_t dxd = vsubq_f32(vld1q_f32(x + j + 12), xi);
      const float32x4_t dyd = vsubq_f32(vld1q_f32(y + j + 12), yi);
      const float32x4_t dzd = vsubq_f32(vld1q_f32(z + j + 12), zi);

      const float32x4_t r2a =
          vfmaq_f32(vfmaq_f32(vfmaq_f32(veps2, dxa, dxa), dya, dya), dza, dza);
      const float32x4_t r2b =
          vfmaq_f32(vfmaq_f32(vfmaq_f32(veps2, dxb, dxb), dyb, dyb), dzb, dzb);
      const float32x4_t r2c =
          vfmaq_f32(vfmaq_f32(vfmaq_f32(veps2, dxc, dxc), dyc, dyc), dzc, dzc);
      const float32x4_t r2d =
          vfmaq_f32(vfmaq_f32(vfmaq_f32(veps2, dxd, dxd), dyd, dyd), dzd, dzd);

      const float32x4_t inva = rsqrt_nr2(r2a);
      const float32x4_t invb = rsqrt_nr2(r2b);
      const float32x4_t invc = rsqrt_nr2(r2c);
      const float32x4_t invd = rsqrt_nr2(r2d);
      const float32x4_t sa =
          vmulq_f32(vld1q_f32(m + j), vmulq_f32(vmulq_f32(inva, inva), inva));
      const float32x4_t sb = vmulq_f32(
          vld1q_f32(m + j + 4), vmulq_f32(vmulq_f32(invb, invb), invb));
      const float32x4_t sc = vmulq_f32(
          vld1q_f32(m + j + 8), vmulq_f32(vmulq_f32(invc, invc), invc));
      const float32x4_t sd = vmulq_f32(
          vld1q_f32(m + j + 12), vmulq_f32(vmulq_f32(invd, invd), invd));

      ax0 = vfmaq_f32(ax0, sa, dxa);
      ay0 = vfmaq_f32(ay0, sa, dya);
      az0 = vfmaq_f32(az0, sa, dza);
      ax1 = vfmaq_f32(ax1, sb, dxb);
      ay1 = vfmaq_f32(ay1, sb, dyb);
      az1 = vfmaq_f32(az1, sb, dzb);
      ax2 = vfmaq_f32(ax2, sc, dxc);
      ay2 = vfmaq_f32(ay2, sc, dyc);
      az2 = vfmaq_f32(az2, sc, dzc);
      ax3 = vfmaq_f32(ax3, sd, dxd);
      ay3 = vfmaq_f32(ay3, sd, dyd);
      az3 = vfmaq_f32(az3, sd, dzd);
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

    float ax = vaddvq_f32(vaddq_f32(vaddq_f32(ax0, ax1), vaddq_f32(ax2, ax3)));
    float ay = vaddvq_f32(vaddq_f32(vaddq_f32(ay0, ay1), vaddq_f32(ay2, ay3)));
    float az = vaddvq_f32(vaddq_f32(vaddq_f32(az0, az1), vaddq_f32(az2, az3)));

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
