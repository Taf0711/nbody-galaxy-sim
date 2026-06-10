#include "nbody/diagnostics.h"

#include <cmath>

namespace nbody {

double kinetic_energy(const Bodies& b) {
  double ke = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double v2 = static_cast<double>(b.vx[i]) * b.vx[i] +
                      static_cast<double>(b.vy[i]) * b.vy[i] +
                      static_cast<double>(b.vz[i]) * b.vz[i];
    ke += 0.5 * b.m[i] * v2;
  }
  return ke;
}

double potential_energy(const Bodies& b, float eps2) {
  const std::size_t n = b.size();
  double pe = 0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double dx = static_cast<double>(b.x[j]) - b.x[i];
      const double dy = static_cast<double>(b.y[j]) - b.y[i];
      const double dz = static_cast<double>(b.z[j]) - b.z[i];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
      pe -= static_cast<double>(b.m[i]) * b.m[j] / r;
    }
  }
  return pe;
}

double total_energy(const Bodies& b, float eps2) {
  return kinetic_energy(b) + potential_energy(b, eps2);
}

std::array<double, 3> momentum(const Bodies& b) {
  std::array<double, 3> p{0, 0, 0};
  for (std::size_t i = 0; i < b.size(); ++i) {
    p[0] += static_cast<double>(b.m[i]) * b.vx[i];
    p[1] += static_cast<double>(b.m[i]) * b.vy[i];
    p[2] += static_cast<double>(b.m[i]) * b.vz[i];
  }
  return p;
}

}  // namespace nbody
