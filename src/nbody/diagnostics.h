#pragma once
#include <array>

#include "nbody/bodies.h"

namespace nbody {

// Conservation diagnostics for tests and validation. Accumulate in double so
// the diagnostic itself doesn't drown the drift it is measuring. The
// potential is the softened pair sum, consistent with the force kernels, so
// total energy is exactly conserved by the continuous dynamics.

double kinetic_energy(const Bodies& b);
double potential_energy(const Bodies& b, float eps2);  // O(n^2)
double total_energy(const Bodies& b, float eps2);
std::array<double, 3> momentum(const Bodies& b);

}  // namespace nbody
