// Physics test suite. Dependency-free: each test prints PASS/FAIL and the
// binary exits nonzero if anything failed.
//
// The two load-bearing ideas:
//  * Correctness is anchored to analytically known behavior (circular
//    two-body orbit, virial-equilibrium Plummer sphere, conservation laws),
//    not to golden files.
//  * Every optimized kernel is validated against the naive one — including
//    Barnes-Hut at theta = 0, which must reproduce the exact all-pairs answer
//    and therefore catches any body lost or double-counted by the tree.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "nbody/barnes_hut.h"
#include "nbody/diagnostics.h"
#include "nbody/forces.h"
#include "nbody/ic.h"
#include "nbody/integrator.h"
#include "nbody/thread_pool.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& name, const std::string& detail) {
  std::printf("[%s] %-38s %s\n", ok ? "PASS" : "FAIL", name.c_str(),
              detail.c_str());
  if (!ok) ++failures;
}

std::string fmt(const char* f, double a, double b = 0) {
  char buf[128];
  std::snprintf(buf, sizeof buf, f, a, b);
  return buf;
}

// Relative error with a floor in the denominator so bodies whose net force
// nearly cancels (tiny |a|) don't dominate the metric.
struct AccelError {
  double rms, max;
};
AccelError accel_error(const nbody::Bodies& ref, const nbody::Bodies& got) {
  double mean_mag = 0;
  for (std::size_t i = 0; i < ref.size(); ++i)
    mean_mag += std::sqrt(static_cast<double>(ref.ax[i]) * ref.ax[i] +
                          static_cast<double>(ref.ay[i]) * ref.ay[i] +
                          static_cast<double>(ref.az[i]) * ref.az[i]);
  mean_mag /= static_cast<double>(ref.size());

  double sum2 = 0, mx = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double dx = static_cast<double>(got.ax[i]) - ref.ax[i];
    const double dy = static_cast<double>(got.ay[i]) - ref.ay[i];
    const double dz = static_cast<double>(got.az[i]) - ref.az[i];
    const double mag = std::sqrt(static_cast<double>(ref.ax[i]) * ref.ax[i] +
                                 static_cast<double>(ref.ay[i]) * ref.ay[i] +
                                 static_cast<double>(ref.az[i]) * ref.az[i]);
    const double rel =
        std::sqrt(dx * dx + dy * dy + dz * dz) / (mag + 0.01 * mean_mag);
    sum2 += rel * rel;
    mx = std::max(mx, rel);
  }
  return {std::sqrt(sum2 / static_cast<double>(ref.size())), mx};
}

void test_two_body_orbit() {
  nbody::Bodies b = nbody::make_two_body();
  const float eps2 = 1e-8f;  // negligible at separation 2
  const float dt = 0.005f;
  auto forces = [&](nbody::Bodies& bb) { nbody::forces_naive(bb, eps2); };

  forces(b);
  const double e0 = nbody::total_energy(b, eps2);

  // Period is 4*pi; run 5 orbits.
  const int steps = static_cast<int>(5.0 * 4.0 * M_PI / dt);
  double max_rdev = 0;
  for (int s = 0; s < steps; ++s) {
    nbody::step_kdk(b, dt, forces);
    const double r = std::sqrt(static_cast<double>(b.x[0]) * b.x[0] +
                               static_cast<double>(b.y[0]) * b.y[0]);
    max_rdev = std::max(max_rdev, std::abs(r - 1.0));
  }
  const double edrift = std::abs((nbody::total_energy(b, eps2) - e0) / e0);

  check(max_rdev < 1e-2, "two-body orbit stays circular",
        fmt("max |r-1| = %.2e over 5 orbits", max_rdev));
  check(edrift < 1e-3, "two-body energy bounded",
        fmt("|dE/E| = %.2e", edrift));
}

void test_plummer_conservation() {
  nbody::ThreadPool pool;
  nbody::Bodies b = nbody::make_plummer(2000, 42);
  const float eps2 = 0.05f * 0.05f;
  const float dt = 0.005f;
  auto forces = [&](nbody::Bodies& bb) {
    nbody::forces_simd(bb, eps2, &pool);
  };

  forces(b);
  const double e0 = nbody::total_energy(b, eps2);
  for (int s = 0; s < 200; ++s) nbody::step_kdk(b, dt, forces);
  const double e1 = nbody::total_energy(b, eps2);
  const auto p = nbody::momentum(b);
  const double pmag = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);

  check(std::abs((e1 - e0) / e0) < 5e-3, "plummer energy conserved",
        fmt("|dE/E| = %.2e (E0 = %.4f)", std::abs((e1 - e0) / e0), e0));
  check(pmag < 1e-3, "plummer momentum conserved",
        fmt("|P| = %.2e after 200 steps", pmag));
}

void test_simd_matches_naive() {
  // 1537 = odd size, exercises the 8-wide, 4-wide and scalar tail paths.
  nbody::Bodies a = nbody::make_plummer(1537, 7);
  nbody::Bodies c = a;
  const float eps2 = 0.05f * 0.05f;
  nbody::forces_naive(a, eps2);
  nbody::forces_simd(c, eps2);
  const auto err = accel_error(a, c);
  check(err.max < 2e-3, "simd matches naive",
        fmt("rel err rms = %.1e, max = %.1e", err.rms, err.max));
}

void test_threaded_matches_serial() {
  nbody::ThreadPool pool;
  nbody::Bodies a = nbody::make_plummer(1000, 3);
  nbody::Bodies c = a;
  const float eps2 = 0.05f * 0.05f;
  nbody::forces_naive(a, eps2);
  nbody::forces_naive(c, eps2, &pool);
  bool same = true;
  for (std::size_t i = 0; i < a.size(); ++i)
    same = same && a.ax[i] == c.ax[i] && a.ay[i] == c.ay[i] &&
           a.az[i] == c.az[i];
  check(same, "threaded bitwise-matches serial",
        same ? "identical (parallel over i only)" : "MISMATCH");
}

void test_bh_theta0_exact() {
  // theta = 0 forces every node open, so Barnes-Hut degenerates to all-pairs
  // via the leaves. Any partition bug (lost/duplicated body) shows up here.
  nbody::Bodies a = nbody::make_plummer(4096, 11);
  nbody::Bodies c = a;
  const float eps2 = 0.05f * 0.05f;
  nbody::forces_naive(a, eps2);
  nbody::BarnesHut bh(0.0f);
  bh.compute(c, eps2);
  const auto err = accel_error(a, c);
  check(err.max < 1e-4, "barnes-hut theta=0 is exact",
        fmt("rel err rms = %.1e, max = %.1e", err.rms, err.max));
}

void test_bh_accuracy() {
  nbody::Bodies a = nbody::make_plummer(4096, 11);
  nbody::Bodies c = a;
  nbody::Bodies d = a;
  const float eps2 = 0.05f * 0.05f;
  nbody::forces_naive(a, eps2);

  nbody::BarnesHut bh05(0.5f);
  bh05.compute(c, eps2);
  const auto e05 = accel_error(a, c);
  check(e05.rms < 1e-2, "barnes-hut theta=0.5 accuracy",
        fmt("rel err rms = %.1e, max = %.1e", e05.rms, e05.max));

  nbody::BarnesHut bh075(0.75f);
  bh075.compute(d, eps2);
  const auto e075 = accel_error(a, d);
  check(e075.rms < 3e-2, "barnes-hut theta=0.75 accuracy",
        fmt("rel err rms = %.1e, max = %.1e", e075.rms, e075.max));
}

void test_bh_energy() {
  nbody::ThreadPool pool;
  nbody::Bodies b = nbody::make_plummer(4096, 23);
  const float eps2 = 0.05f * 0.05f;
  const float dt = 0.005f;
  nbody::BarnesHut bh(0.7f);
  auto forces = [&](nbody::Bodies& bb) { bh.compute(bb, eps2, &pool); };

  forces(b);
  const double e0 = nbody::total_energy(b, eps2);
  for (int s = 0; s < 100; ++s) nbody::step_kdk(b, dt, forces);
  const double drift = std::abs((nbody::total_energy(b, eps2) - e0) / e0);
  check(drift < 1e-2, "barnes-hut energy conserved",
        fmt("|dE/E| = %.2e over 100 steps (theta=0.7)", drift));
}

}  // namespace

int main() {
  test_two_body_orbit();
  test_plummer_conservation();
  test_simd_matches_naive();
  test_threaded_matches_serial();
  test_bh_theta0_exact();
  test_bh_accuracy();
  test_bh_energy();
  std::printf("\n%s (%d failure%s)\n", failures == 0 ? "OK" : "FAILED",
              failures, failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
