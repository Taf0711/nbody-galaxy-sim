#include "nbody/ic.h"

#include <cmath>
#include <random>

namespace nbody {

Bodies make_two_body() {
  // Each body circles the COM at r = 1: F = G*m^2/d^2 = 1/4, and
  // m*v^2/r = F gives v = 0.5.
  Bodies b;
  b.resize(2);
  b.m[0] = b.m[1] = 1.0f;
  b.x[0] = 1.0f;
  b.x[1] = -1.0f;
  b.vy[0] = 0.5f;
  b.vy[1] = -0.5f;
  return b;
}

Bodies make_plummer(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> U(0.0f, 1.0f);

  Bodies b;
  b.resize(n);
  const float mi = 1.0f / static_cast<float>(n);

  for (std::size_t i = 0; i < n; ++i) {
    // Radius from the cumulative mass profile M(r)/M = r^3 / (1 + r^2)^(3/2),
    // capped at 8a so a handful of far outliers don't dominate the box.
    float r;
    do {
      float u = std::max(U(rng), 1e-7f);
      r = 1.0f / std::sqrt(std::pow(u, -2.0f / 3.0f) - 1.0f);
    } while (r > 8.0f);

    auto sphere_dir = [&](float& dx, float& dy, float& dz) {
      float cz = 2.0f * U(rng) - 1.0f;
      float phi = 6.2831853f * U(rng);
      float s = std::sqrt(std::max(0.0f, 1.0f - cz * cz));
      dx = s * std::cos(phi);
      dy = s * std::sin(phi);
      dz = cz;
    };

    float dx, dy, dz;
    sphere_dir(dx, dy, dz);
    b.x[i] = r * dx;
    b.y[i] = r * dy;
    b.z[i] = r * dz;

    // Speed: fraction q of the local escape velocity, q sampled by rejection
    // from g(q) ~ q^2 (1 - q^2)^(7/2) (max ~0.092 at q^2 = 2/9).
    float q, gq;
    do {
      q = U(rng);
      gq = 0.1f * U(rng);
    } while (gq > q * q * std::pow(1.0f - q * q, 3.5f));
    const float vesc = std::sqrt(2.0f) * std::pow(1.0f + r * r, -0.25f);
    const float v = q * vesc;

    sphere_dir(dx, dy, dz);
    b.vx[i] = v * dx;
    b.vy[i] = v * dy;
    b.vz[i] = v * dz;

    b.m[i] = mi;
  }

  // Recenter so tests can assert conservation against an exact zero.
  double cx = 0, cy = 0, cz = 0, px = 0, py = 0, pz = 0, M = 0;
  for (std::size_t i = 0; i < n; ++i) {
    M += b.m[i];
    cx += static_cast<double>(b.m[i]) * b.x[i];
    cy += static_cast<double>(b.m[i]) * b.y[i];
    cz += static_cast<double>(b.m[i]) * b.z[i];
    px += static_cast<double>(b.m[i]) * b.vx[i];
    py += static_cast<double>(b.m[i]) * b.vy[i];
    pz += static_cast<double>(b.m[i]) * b.vz[i];
  }
  for (std::size_t i = 0; i < n; ++i) {
    b.x[i] -= static_cast<float>(cx / M);
    b.y[i] -= static_cast<float>(cy / M);
    b.z[i] -= static_cast<float>(cz / M);
    b.vx[i] -= static_cast<float>(px / M);
    b.vy[i] -= static_cast<float>(py / M);
    b.vz[i] -= static_cast<float>(pz / M);
  }
  return b;
}

}  // namespace nbody
