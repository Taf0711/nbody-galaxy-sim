#pragma once
#include <cstdint>
#include <utility>
#include <vector>

#include "nbody/bodies.h"

namespace nbody {

class ThreadPool;

// Barnes-Hut O(n log n) approximate forces.
//
// Design: instead of a pointer-chasing insertion octree, bodies are sorted by
// 63-bit Morton (Z-order) code each step and the tree is built by recursively
// splitting code ranges on 3-bit digits. Consequences:
//   * every node's bodies form a contiguous range of the sorted arrays, so
//     leaf interactions run through the same NEON kernel shape as the
//     brute-force path instead of gathering scattered bodies;
//   * the tree is a flat std::vector<Node> with children stored contiguously
//     (parents always precede children), so center-of-mass aggregation is a
//     single reverse sweep, no recursion;
//   * traversing bodies in Morton order gives consecutive bodies nearly
//     identical tree walks, which is what makes the parallel traversal cache-
//     friendly.
//
// The tree is rebuilt from scratch every step (no incremental updates);
// buffers persist across rebuilds so steady-state allocation is zero.
class BarnesHut {
 public:
  // quadrupole: store second moments per node and add the quadrupole term
  // when a node is accepted. Costs ~3x the flops per accepted node but cuts
  // the error at fixed theta ~4x (one extra power of s/d), so theta can be
  // opened up: quad at theta=0.65 matches monopole theta=0.5 accuracy while
  // accepting ~2.2x fewer nodes.
  explicit BarnesHut(float theta = 0.5f, bool quadrupole = false)
      : theta_(theta), quad_(quadrupole) {}

  // Writes accelerations into b.ax/ay/az, same contract as forces_naive.
  void compute(Bodies& b, float eps2, ThreadPool* pool = nullptr);

  float theta() const { return theta_; }
  double last_build_ms() const { return build_ms_; }
  double last_traverse_ms() const { return traverse_ms_; }
  std::size_t node_count() const { return nodes_.size(); }

 private:
  struct Node {
    float comx, comy, comz, mass;  // center of mass
    float size;                    // cell side length (acceptance criterion)
    std::uint32_t first, count;    // body range in Morton-sorted order
    std::uint32_t first_child;     // 0 == leaf (node 0 is the root, never a child)
    std::uint32_t nchild;
    // Traceless quadrupole about the COM: Q_ij = sum m (3 s_i s_j - s^2 d_ij).
    float qxx, qyy, qzz, qxy, qxz, qyz;
  };

  void split(std::uint32_t idx, int shift);
  void accel_range(Bodies& b, float eps2, std::size_t k0, std::size_t k1) const;

  float theta_;
  bool quad_;
  double build_ms_ = 0, traverse_ms_ = 0;

  // (morton code, original index), sorted by code each step.
  std::vector<std::pair<std::uint64_t, std::uint32_t>> keys_;
  std::vector<std::uint64_t> scodes_;          // sorted codes (split lookups)
  std::vector<float> sx_, sy_, sz_, sm_;       // bodies permuted to sort order
  std::vector<Node> nodes_;
};

}  // namespace nbody
