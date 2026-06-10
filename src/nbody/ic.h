#pragma once
#include <cstdint>

#include "nbody/bodies.h"

namespace nbody {

// Two unit masses on a circular orbit of radius 1 about the origin in the
// xy-plane (G = 1, separation 2, v = 0.5, period 4*pi). Analytic reference
// case for integrator tests.
Bodies make_two_body();

// Plummer sphere in virial equilibrium: M = 1, scale radius a = 1, isotropic
// velocities sampled from the exact distribution function (Aarseth's
// rejection method). Centered to zero net position and momentum.
Bodies make_plummer(std::size_t n, std::uint64_t seed = 1);

}  // namespace nbody
