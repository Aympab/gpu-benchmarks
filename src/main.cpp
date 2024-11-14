#include "func.hpp"
#include <hipSYCL/sycl/usm.hpp>

using bm_vec_t = std::vector<int64_t>;
static bm_vec_t N1_RANGE = {1024, 2048, 4096};
static bm_vec_t N2_RANGE = {128, 256, 1024};

// ================================================
// ================================================
static void BM_GlobalMem(benchmark::State &state) {
  sycl::queue q;
  Params params(state.range(0), state.range(1));

  auto n1 = params.n1_;
  auto n2 = params.n2_;

  /* Benchmark infos */
  state.counters.insert({
      {"n1", n1},
      {"n2", n2},
      {"w1", 1},
      {"w2", W},
  });

  /* Data setup */
  auto buffer = sycl::malloc_shared<real_t>(n1 * n2, q);
  fill(q, buffer, params).wait();
  auto scratch = sycl::malloc_shared<real_t>(n1 * n2, q);
  q.wait();

  /* Benchmark */
  for (auto _ : state) {
    try {
      solveGlobal(q, buffer, scratch, params).wait();
    } catch (const sycl::exception &e) {
      state.SkipWithError(e.what());
      break;
    }
  }

  auto err = validate(q, buffer, params);
  validate_bench(err, state);

  auto nIter = state.iterations();

  state.SetItemsProcessed(nIter * n1 * n2);
  state.SetBytesProcessed(nIter * n1 * n2 * sizeof(real_t));
  state.counters.insert({{"nIter", nIter}});
  state.counters.insert({{"err", err}});

  sycl::free(buffer, q);
  sycl::free(scratch, q);
} // end BM_GlobalMem

// ================================================
BENCHMARK(BM_GlobalMem)
    ->ArgsProduct({
        N1_RANGE, /*n1*/
        N2_RANGE, /*n2*/
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ================================================
// ================================================
static void BM_LocalMem(benchmark::State &state) {
  sycl::queue q;
  Params params(state.range(0), state.range(1));
  auto w1 = state.range(2);
  auto w2 = W/w1;

  auto n1 = params.n1_;
  auto n2 = params.n2_;

  /* Benchmark infos */
  state.counters.insert({
      {"n1", n1},
      {"n2", n2},
      {"w1", w1},
      {"w2", w2},
  });

  /* Data setup */
  auto buffer = sycl::malloc_shared<real_t>(n1 * n2, q);
  fill(q, buffer, params).wait();
  q.wait();

  /* Benchmark */
  for (auto _ : state) {
    try {
      solveLocal(q, buffer, params, w1, w2).wait();
    } catch (const sycl::exception &e) {
      state.SkipWithError(e.what());
      break;
    }
  }

  auto err = validate(q, buffer, params);
  validate_bench(err, state);

  auto nIter = state.iterations();

  state.SetItemsProcessed(nIter * n1 * n2);
  state.SetBytesProcessed(nIter * n1 * n2 * sizeof(real_t));
  state.counters.insert({{"nIter", nIter}});
  state.counters.insert({{"err", err}});

  sycl::free(buffer, q);
} // end BM_LocalMem

// ================================================
BENCHMARK(BM_LocalMem)
    ->ArgsProduct({
        N1_RANGE, /*n1*/
        N2_RANGE, /*n2*/
        {1, 2, 4, 8, 16},          /*w1*/
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();