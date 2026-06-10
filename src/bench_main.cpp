// Benchmark harness. Produces the numbers behind the README charts.
//
//   ./build/nbody_bench [--quick] [--csv path] [--no-scaling]
//
// Two sections:
//   sweep    ms/step for each method across N (the headline chart)
//   scaling  SIMD+threads at fixed N across thread counts (P/E core story)
//
// Protocol per configuration: build a fresh Plummer sphere (fixed seed), run
// one untimed priming step (warms caches, faults pages, spins up threads),
// then time individual steps and report the median. Reps adapt to step cost
// so slow configs don't take minutes and fast ones aren't one-sample noise.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "nbody/barnes_hut.h"
#include "nbody/forces.h"
#include "nbody/ic.h"
#include "nbody/integrator.h"
#include "nbody/thread_pool.h"

namespace {

struct Row {
  std::string section, method;
  std::size_t n;
  unsigned threads;
  float theta;        // 0 = n/a
  double ms_step;
  double build_ms;    // Barnes-Hut only
  double traverse_ms; // Barnes-Hut only
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Returns median ms/step. `bh` non-null when the force fn wraps a BarnesHut,
// so per-phase timings can be captured alongside.
double time_method(std::size_t n, const nbody::ForceFn& forces,
                   nbody::BarnesHut* bh, double* build_ms,
                   double* traverse_ms, bool quick) {
  nbody::Bodies b = nbody::make_plummer(n, 42);
  const float dt = 0.005f;

  forces(b);  // prime accelerations + caches

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  nbody::step_kdk(b, dt, forces);
  const double probe =
      std::chrono::duration<double, std::milli>(clock::now() - t0).count();

  const double budget_ms = quick ? 400.0 : 1500.0;
  const int reps = std::clamp(static_cast<int>(budget_ms / std::max(probe, 0.01)),
                              3, quick ? 10 : 40);

  std::vector<double> times, builds, traverses;
  times.reserve(reps);
  for (int r = 0; r < reps; ++r) {
    t0 = clock::now();
    nbody::step_kdk(b, dt, forces);
    times.push_back(
        std::chrono::duration<double, std::milli>(clock::now() - t0).count());
    if (bh) {
      builds.push_back(bh->last_build_ms());
      traverses.push_back(bh->last_traverse_ms());
    }
  }
  if (bh) {
    *build_ms = median(builds);
    *traverse_ms = median(traverses);
  }
  return median(times);
}

void print_row(const Row& r) {
  std::printf("%-12s n=%8zu t=%2u  %10.3f ms/step", r.method.c_str(), r.n,
              r.threads, r.ms_step);
  if (r.theta > 0)
    std::printf("   (theta=%.2f, build %.2f ms + traverse %.2f ms)", r.theta,
                r.build_ms, r.traverse_ms);
  std::printf("\n");
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false, scaling = true;
  std::string csv_path = "bench/results.csv";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--quick")) quick = true;
    else if (!std::strcmp(argv[i], "--no-scaling")) scaling = false;
    else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
    else {
      std::fprintf(stderr,
                   "usage: %s [--quick] [--csv path] [--no-scaling]\n",
                   argv[0]);
      return 2;
    }
  }

  const unsigned ncpu = std::thread::hardware_concurrency();
  nbody::ThreadPool pool(ncpu);
  std::printf("nbody_bench: %u hardware threads%s\n\n", ncpu,
              quick ? " (quick mode)" : "");

  std::vector<Row> rows;

  struct Case {
    const char* method;
    std::vector<std::size_t> sizes;
    float theta;
  };
  // Sizes chosen so each config finishes in seconds while the O(n^2) vs
  // O(n log n) divergence is unmistakable.
  const std::vector<Case> cases = {
      {"naive", {1024, 4096, 16384}, 0},
      {"naive_mt", {4096, 16384, 65536}, 0},
      {"simd", {1024, 4096, 16384, 65536}, 0},
      {"simd_mt", {4096, 16384, 65536, 131072}, 0},
      {"bh_mt", {16384, 65536, 262144, 1048576}, 0.5f},
  };

  std::printf("== sweep ==\n");
  for (const auto& c : cases) {
    for (std::size_t n : c.sizes) {
      if (quick && n > 16384) continue;
      Row r{"sweep", c.method, n, 1, c.theta, 0, 0, 0};
      const std::string m = c.method;
      if (m == "naive") {
        r.ms_step = time_method(
            n, [&](nbody::Bodies& b) { nbody::forces_naive(b, 0.0025f); },
            nullptr, nullptr, nullptr, quick);
      } else if (m == "naive_mt") {
        r.threads = ncpu;
        r.ms_step = time_method(
            n,
            [&](nbody::Bodies& b) { nbody::forces_naive(b, 0.0025f, &pool); },
            nullptr, nullptr, nullptr, quick);
      } else if (m == "simd") {
        r.ms_step = time_method(
            n, [&](nbody::Bodies& b) { nbody::forces_simd(b, 0.0025f); },
            nullptr, nullptr, nullptr, quick);
      } else if (m == "simd_mt") {
        r.threads = ncpu;
        r.ms_step = time_method(
            n,
            [&](nbody::Bodies& b) { nbody::forces_simd(b, 0.0025f, &pool); },
            nullptr, nullptr, nullptr, quick);
      } else {  // bh_mt
        r.threads = ncpu;
        nbody::BarnesHut bh(c.theta);
        r.ms_step = time_method(
            n,
            [&](nbody::Bodies& b) { bh.compute(b, 0.0025f, &pool); }, &bh,
            &r.build_ms, &r.traverse_ms, quick);
      }
      print_row(r);
      rows.push_back(r);
    }
  }

  if (scaling && !quick) {
    std::printf("\n== thread scaling (simd, n=32768) ==\n");
    std::vector<unsigned> tcounts;
    for (unsigned t = 1; t <= ncpu; t == 1 ? t = 2 : t += 2) tcounts.push_back(t);
    if (tcounts.back() != ncpu) tcounts.push_back(ncpu);
    for (unsigned t : tcounts) {
      nbody::ThreadPool p(t);
      Row r{"scaling", "simd_mt", 32768, t, 0, 0, 0, 0};
      r.ms_step = time_method(
          32768,
          [&](nbody::Bodies& b) { nbody::forces_simd(b, 0.0025f, &p); },
          nullptr, nullptr, nullptr, quick);
      print_row(r);
      rows.push_back(r);
    }
  }

  std::filesystem::create_directories(
      std::filesystem::path(csv_path).parent_path());
  if (FILE* f = std::fopen(csv_path.c_str(), "w")) {
    std::fprintf(f, "section,method,n,threads,theta,ms_step,build_ms,traverse_ms\n");
    for (const auto& r : rows)
      std::fprintf(f, "%s,%s,%zu,%u,%.2f,%.4f,%.4f,%.4f\n", r.section.c_str(),
                   r.method.c_str(), r.n, r.threads, r.theta, r.ms_step,
                   r.build_ms, r.traverse_ms);
    std::fclose(f);
    std::printf("\nwrote %s\n", csv_path.c_str());
  }
  return 0;
}
