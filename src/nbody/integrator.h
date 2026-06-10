#pragma once
#include <functional>

#include "nbody/bodies.h"

namespace nbody {

using ForceFn = std::function<void(Bodies&)>;

inline void kick(Bodies& b, float h) {
  const std::size_t n = b.size();
  for (std::size_t i = 0; i < n; ++i) {
    b.vx[i] += h * b.ax[i];
    b.vy[i] += h * b.ay[i];
    b.vz[i] += h * b.az[i];
  }
}

inline void drift(Bodies& b, float h) {
  const std::size_t n = b.size();
  for (std::size_t i = 0; i < n; ++i) {
    b.x[i] += h * b.vx[i];
    b.y[i] += h * b.vy[i];
    b.z[i] += h * b.vz[i];
  }
}

// One kick-drift-kick leapfrog step. Symplectic: energy error stays bounded
// instead of accumulating secularly. Requires b.ax/ay/az to hold the
// accelerations for the *current* positions, so call `forces` once before the
// first step; thereafter the trailing half-kick's force evaluation is reused.
inline void step_kdk(Bodies& b, float dt, const ForceFn& forces) {
  kick(b, 0.5f * dt);
  drift(b, dt);
  forces(b);
  kick(b, 0.5f * dt);
}

}  // namespace nbody
