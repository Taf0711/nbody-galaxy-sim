#pragma once
#include <cstddef>
#include <vector>

namespace nbody {

// Structure-of-arrays particle store. SoA keeps each coordinate contiguous so
// SIMD kernels can load 4 bodies per instruction; an AoS layout would need a
// gather for every lane.
struct Bodies {
  std::vector<float> x, y, z;
  std::vector<float> vx, vy, vz;
  std::vector<float> ax, ay, az;
  std::vector<float> m;

  std::size_t size() const { return x.size(); }

  void resize(std::size_t n) {
    for (auto* v : {&x, &y, &z, &vx, &vy, &vz, &ax, &ay, &az, &m})
      v->resize(n, 0.0f);
  }
};

}  // namespace nbody
