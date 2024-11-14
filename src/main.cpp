#include "func.hpp"
#include <hipSYCL/sycl/usm.hpp>

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
  });

  /* Data setup */
  auto buffer = sycl::malloc_device<real_t>(n1 * n2, q);
  fill(q, buffer, params).wait();
  auto scratch = sycl::malloc_device<real_t>(n1 * n2, q);

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
} // end BM_LocalMem

// ================================================
// ================================================
static void BM_LocalMem(benchmark::State &state) {
  sycl::queue q;
  Params params(state.range(0), state.range(1));

  auto n1 = params.n1_;
  auto n2 = params.n2_;

  /* Benchmark infos */
  state.counters.insert({
      {"n1", n1},
      {"n2", n2},
  });

  /* Data setup */
  auto buffer = sycl::malloc_device<real_t>(n1 * n2, q);
  fill(q, buffer, params).wait();

  /* Benchmark */
  for (auto _ : state) {
    try {
      solveLocal(q, buffer, params).wait();
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
        {4096, 8192}, /*n1*/
        {128},        /*n2*/
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();