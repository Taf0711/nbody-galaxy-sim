#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <pthread/qos.h>
#endif

namespace nbody {

// Fixed pool that executes chunked parallel-for jobs. A pool of size T spawns
// T-1 workers; the calling thread participates in every job, so ThreadPool(1)
// degenerates to a plain serial loop with no synchronization.
//
// Work is distributed dynamically: threads pull `grain`-sized chunks off a
// shared atomic cursor, so uneven per-element cost (e.g. Barnes-Hut traversal
// depth) balances out without static partitioning.
class ThreadPool {
 public:
  explicit ThreadPool(unsigned threads = std::thread::hardware_concurrency())
      : nthreads_(threads == 0 ? 1 : threads) {
    for (unsigned i = 0; i + 1 < nthreads_; ++i)
      workers_.emplace_back([this] { worker_loop(); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_start_.notify_all();
    for (auto& w : workers_) w.join();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned size() const { return nthreads_; }

  // Runs fn(lo, hi) over [begin, end) split into chunks of `grain`. Blocks
  // until the whole range is done. Not reentrant.
  void parallel_for(std::size_t begin, std::size_t end, std::size_t grain,
                    const std::function<void(std::size_t, std::size_t)>& fn) {
    if (begin >= end) return;
    if (workers_.empty() || end - begin <= grain) {
      fn(begin, end);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      job_ = &fn;
      next_.store(begin, std::memory_order_relaxed);
      end_ = end;
      grain_ = grain == 0 ? 1 : grain;
      pending_ = static_cast<unsigned>(workers_.size());
      ++epoch_;
    }
    cv_start_.notify_all();
    run_chunks();
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [this] { return pending_ == 0; });
    job_ = nullptr;
  }

 private:
  void worker_loop() {
#ifdef __APPLE__
    // On Apple Silicon, thread QoS decides P-core vs E-core eligibility.
    // Workers inheriting an unfavorable class get parked on E-cores and the
    // pool silently scales ~2x instead of ~5x; pin them to a class the
    // scheduler will put on performance cores.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_start_.wait(lk, [&] { return stop_ || epoch_ != seen; });
        if (stop_) return;
        seen = epoch_;
      }
      run_chunks();
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (--pending_ == 0) cv_done_.notify_one();
      }
    }
  }

  void run_chunks() {
    const auto& fn = *job_;
    for (;;) {
      std::size_t lo = next_.fetch_add(grain_, std::memory_order_relaxed);
      if (lo >= end_) break;
      fn(lo, std::min(lo + grain_, end_));
    }
  }

  unsigned nthreads_;
  std::vector<std::thread> workers_;

  std::mutex mu_;
  std::condition_variable cv_start_, cv_done_;
  const std::function<void(std::size_t, std::size_t)>* job_ = nullptr;
  std::atomic<std::size_t> next_{0};
  std::size_t end_ = 0;
  std::size_t grain_ = 1;
  std::uint64_t epoch_ = 0;
  unsigned pending_ = 0;
  bool stop_ = false;
};

}  // namespace nbody
