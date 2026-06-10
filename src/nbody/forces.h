#pragma once
#include "nbody/bodies.h"

namespace nbody {

class ThreadPool;

// Force kernels write softened gravitational accelerations (G = 1) into
// b.ax/ay/az:
//
//   a_i = sum_j m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps2)^(3/2)
//
// eps2 must be > 0. With Plummer softening the i == j term contributes
// exactly zero (numerator is the zero vector, denominator is eps^3), so no
// kernel needs an i != j branch — which is what makes the inner loops
// branch-free and vectorizable.
//
// Passing a pool parallelizes over i; per-i results are bitwise identical to
// the serial run because each row's j-summation order is unchanged.

void forces_naive(Bodies& b, float eps2, ThreadPool* pool = nullptr);
void forces_simd(Bodies& b, float eps2, ThreadPool* pool = nullptr);

}  // namespace nbody
