#include "nbody/barnes_hut.h"

#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>

#include "nbody/thread_pool.h"

namespace nbody {
namespace {

constexpr std::uint32_t kLeafCap = 16;
constexpr int kMortonBits = 21;  // per axis; 63-bit codes, max tree depth 21

// Spread the low 21 bits of v so there are two zero bits between each
// (standard magic-mask sequence, cf. libmorton).
inline std::uint64_t expand_bits21(std::uint64_t v) {
  v &= 0x1fffff;
  v = (v | (v << 32)) & 0x1f00000000ffffULL;
  v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
  v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
  v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
  v = (v | (v << 2)) & 0x1249249249249249ULL;
  return v;
}

inline std::uint64_t morton3(std::uint32_t ix, std::uint32_t iy,
                             std::uint32_t iz) {
  return (expand_bits21(ix) << 2) | (expand_bits21(iy) << 1) | expand_bits21(iz);
}

inline float32x4_t rsqrt_nr2(float32x4_t v) {
  float32x4_t e = vrsqrteq_f32(v);
  e = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
  e = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
  return e;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

}  // namespace

void BarnesHut::split(std::uint32_t idx, int shift) {
  const std::uint32_t first = nodes_[idx].first;
  const std::uint32_t count = nodes_[idx].count;
  // Depth exhaustion (shift < 0) happens when > kLeafCap bodies share a
  // Morton cell — coincident positions. They become one big leaf; direct
  // summation there is still exact.
  if (count <= kLeafCap || shift < 0) return;

  const float child_size = nodes_[idx].size * 0.5f;
  std::uint32_t lo[8], cnt[8];
  int nc = 0;
  std::uint32_t pos = first;
  const std::uint32_t hi_all = first + count;
  for (std::uint32_t d = 0; d < 8 && pos < hi_all; ++d) {
    // Codes are sorted, so each octant digit owns a contiguous subrange.
    auto it = std::upper_bound(
        scodes_.begin() + pos, scodes_.begin() + hi_all, d,
        [shift](std::uint32_t val, std::uint64_t code) {
          return val < ((code >> shift) & 7u);
        });
    const auto hi = static_cast<std::uint32_t>(it - scodes_.begin());
    if (hi > pos) {
      lo[nc] = pos;
      cnt[nc] = hi - pos;
      ++nc;
    }
    pos = hi;
  }

  const auto first_child = static_cast<std::uint32_t>(nodes_.size());
  nodes_[idx].first_child = first_child;
  nodes_[idx].nchild = static_cast<std::uint32_t>(nc);
  for (int c = 0; c < nc; ++c) {
    Node nd{};
    nd.first = lo[c];
    nd.count = cnt[c];
    nd.size = child_size;
    nodes_.push_back(nd);
  }
  for (int c = 0; c < nc; ++c) split(first_child + c, shift - 3);
}

void BarnesHut::compute(Bodies& b, float eps2, ThreadPool* pool) {
  const std::size_t n = b.size();
  if (n == 0) return;
  const auto t0 = std::chrono::steady_clock::now();

  // 1. Bounding cube.
  float mnx = b.x[0], mny = b.y[0], mnz = b.z[0];
  float mxx = mnx, mxy = mny, mxz = mnz;
  for (std::size_t i = 1; i < n; ++i) {
    mnx = std::min(mnx, b.x[i]);
    mny = std::min(mny, b.y[i]);
    mnz = std::min(mnz, b.z[i]);
    mxx = std::max(mxx, b.x[i]);
    mxy = std::max(mxy, b.y[i]);
    mxz = std::max(mxz, b.z[i]);
  }
  const float side =
      std::max({mxx - mnx, mxy - mny, mxz - mnz, 1e-6f}) * 1.0001f;

  // 2. Morton codes, sorted with the original index carried along.
  keys_.resize(n);
  const float scale = static_cast<float>(1u << kMortonBits) / side;
  const float qmax = static_cast<float>((1u << kMortonBits) - 1);
  for (std::size_t i = 0; i < n; ++i) {
    const auto qx = static_cast<std::uint32_t>(
        std::clamp((b.x[i] - mnx) * scale, 0.0f, qmax));
    const auto qy = static_cast<std::uint32_t>(
        std::clamp((b.y[i] - mny) * scale, 0.0f, qmax));
    const auto qz = static_cast<std::uint32_t>(
        std::clamp((b.z[i] - mnz) * scale, 0.0f, qmax));
    keys_[i] = {morton3(qx, qy, qz), static_cast<std::uint32_t>(i)};
  }
  std::sort(keys_.begin(), keys_.end());

  // 3. Permute bodies into Morton order so leaves are contiguous.
  scodes_.resize(n);
  sx_.resize(n);
  sy_.resize(n);
  sz_.resize(n);
  sm_.resize(n);
  for (std::size_t k = 0; k < n; ++k) {
    scodes_[k] = keys_[k].first;
    const std::uint32_t i = keys_[k].second;
    sx_[k] = b.x[i];
    sy_[k] = b.y[i];
    sz_[k] = b.z[i];
    sm_[k] = b.m[i];
  }

  // 4. Build the tree top-down by splitting code ranges.
  nodes_.clear();
  if (nodes_.capacity() == 0) nodes_.reserve(n / 2);
  Node root{};
  root.first = 0;
  root.count = static_cast<std::uint32_t>(n);
  root.size = side;
  nodes_.push_back(root);
  split(0, 3 * (kMortonBits - 1));  // top octant digit lives at bits 60..62

  // 5. Centers of mass: children always follow parents, so one reverse sweep
  // aggregates leaves before the nodes that contain them.
  for (std::size_t k = nodes_.size(); k-- > 0;) {
    Node& nd = nodes_[k];
    float M = 0, cx = 0, cy = 0, cz = 0;
    if (nd.first_child == 0) {
      for (std::uint32_t i = nd.first; i < nd.first + nd.count; ++i) {
        M += sm_[i];
        cx += sm_[i] * sx_[i];
        cy += sm_[i] * sy_[i];
        cz += sm_[i] * sz_[i];
      }
    } else {
      for (std::uint32_t c = 0; c < nd.nchild; ++c) {
        const Node& ch = nodes_[nd.first_child + c];
        M += ch.mass;
        cx += ch.mass * ch.comx;
        cy += ch.mass * ch.comy;
        cz += ch.mass * ch.comz;
      }
    }
    nd.mass = M;
    const float invM = M > 0 ? 1.0f / M : 0.0f;
    nd.comx = cx * invM;
    nd.comy = cy * invM;
    nd.comz = cz * invM;
  }
  build_ms_ = ms_since(t0);

  // 6. Traversal, parallel over bodies in Morton order.
  const auto t1 = std::chrono::steady_clock::now();
  if (pool) {
    pool->parallel_for(0, n, 128, [&](std::size_t k0, std::size_t k1) {
      accel_range(b, eps2, k0, k1);
    });
  } else {
    accel_range(b, eps2, 0, n);
  }
  traverse_ms_ = ms_since(t1);
}

void BarnesHut::accel_range(Bodies& b, float eps2, std::size_t k0,
                            std::size_t k1) const {
  const float theta2 = theta_ * theta_;
  const float* sx = sx_.data();
  const float* sy = sy_.data();
  const float* sz = sz_.data();
  const float* sm = sm_.data();

  for (std::size_t k = k0; k < k1; ++k) {
    const float xi = sx[k], yi = sy[k], zi = sz[k];
    float ax = 0, ay = 0, az = 0;

    // Worst case is ~depth * 7 + 1 entries; 21 * 7 + 1 = 148.
    std::uint32_t stack[256];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
      const Node& nd = nodes_[stack[--sp]];
      const float dx = nd.comx - xi;
      const float dy = nd.comy - yi;
      const float dz = nd.comz - zi;
      const float d2 = dx * dx + dy * dy + dz * dz;

      // Never approximate a node this body sits inside: its own mass would
      // contribute a spurious self-force through the COM.
      const bool contains_self = k >= nd.first && k < nd.first + nd.count;

      if (!contains_self && nd.size * nd.size < theta2 * d2) {
        const float r2 = d2 + eps2;
        const float inv = 1.0f / std::sqrt(r2);
        const float s = nd.mass * inv * inv * inv;
        ax += s * dx;
        ay += s * dy;
        az += s * dz;
      } else if (nd.first_child == 0) {
        // Leaf: direct sum over its contiguous body range, 4 lanes at a time.
        const std::uint32_t lo = nd.first, hi = nd.first + nd.count;
        std::uint32_t j = lo;
        if (hi - lo >= 4) {
          const float32x4_t xiv = vdupq_n_f32(xi);
          const float32x4_t yiv = vdupq_n_f32(yi);
          const float32x4_t ziv = vdupq_n_f32(zi);
          float32x4_t axv = vdupq_n_f32(0.0f), ayv = axv, azv = axv;
          for (; j + 4 <= hi; j += 4) {
            const float32x4_t ddx = vsubq_f32(vld1q_f32(sx + j), xiv);
            const float32x4_t ddy = vsubq_f32(vld1q_f32(sy + j), yiv);
            const float32x4_t ddz = vsubq_f32(vld1q_f32(sz + j), ziv);
            const float32x4_t r2v = vfmaq_f32(
                vfmaq_f32(vfmaq_f32(vdupq_n_f32(eps2), ddx, ddx), ddy, ddy),
                ddz, ddz);
            const float32x4_t inv = rsqrt_nr2(r2v);
            const float32x4_t s = vmulq_f32(
                vld1q_f32(sm + j), vmulq_f32(vmulq_f32(inv, inv), inv));
            axv = vfmaq_f32(axv, s, ddx);
            ayv = vfmaq_f32(ayv, s, ddy);
            azv = vfmaq_f32(azv, s, ddz);
          }
          ax += vaddvq_f32(axv);
          ay += vaddvq_f32(ayv);
          az += vaddvq_f32(azv);
        }
        for (; j < hi; ++j) {
          const float ddx = sx[j] - xi;
          const float ddy = sy[j] - yi;
          const float ddz = sz[j] - zi;
          const float r2 = ddx * ddx + ddy * ddy + ddz * ddz + eps2;
          const float inv = 1.0f / std::sqrt(r2);
          const float s = sm[j] * inv * inv * inv;
          ax += s * ddx;
          ay += s * ddy;
          az += s * ddz;
        }
      } else {
        for (std::uint32_t c = 0; c < nd.nchild; ++c)
          stack[sp++] = nd.first_child + c;
      }
    }

    const std::uint32_t orig = keys_[k].second;
    b.ax[orig] = ax;
    b.ay[orig] = ay;
    b.az[orig] = az;
  }
}

}  // namespace nbody
