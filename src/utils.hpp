#pragma once

#include <benchmark/benchmark.h>
#include <experimental/mdspan>
#include <sycl/sycl.hpp>

using real_t = double;

using mdspan2d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 2>,
                              std::experimental::layout_right>;

constexpr auto W = 128;

struct Params {
  size_t n1_, n2_;

  Params(const size_t &n1, const size_t &n2) : n1_(n1), n2_(n2) {}
};

//==============================================================================
//==============================================================================
/* Fills the data buffer */
inline sycl::event fill(sycl::queue &q, real_t *buffer, Params &p) {
  auto n1 = p.n1_;
  auto n2 = p.n2_;

  const sycl::range global_size{n1, n2};
  const sycl::range local_size{1, W};

  const sycl::nd_range ndr(global_size, local_size);

  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(ndr, [=](auto itm) {
      auto i = itm.get_global_id(0);
      auto j = itm.get_global_id(1);

      mdspan2d_t data(buffer, n1, n2);

      data(i, j) = i * j;
    }); // end parallel_for
  });   // end q.submit
}

//==============================================================================
//==============================================================================
inline real_t validate(sycl::queue &q, real_t *buffer, Params &p) {
  auto n1 = p.n1_;
  auto n2 = p.n2_;

  const sycl::range global_size{n1, n2};
  const sycl::range local_size{1, W};
  const sycl::nd_range ndr(global_size, local_size);

  real_t errorL1 = 0.0;
  sycl::buffer<real_t> errorL1_buff(&errorL1, 1);

  q.submit([&](sycl::handler &cgh) {
     sycl::accessor errorL1_acc(errorL1_buff, cgh, sycl::read_write);
     auto errorL1_reduc = sycl::reduction(errorL1_acc, sycl::plus<real_t>());
     cgh.parallel_for(ndr, errorL1_reduc, [=](auto itm, auto &err) {
       mdspan2d_t data(buffer, n1, n2);

       auto i = itm.get_global_id(0);
       auto j = itm.get_global_id(1);

       auto value = data(i, j);
       auto expected = sycl::sin(static_cast<real_t>(i + j));

       err += sycl::fabs(value - expected);
     });
   }).wait(); // end q.submit

  return errorL1;
}

//==============================================================================
//==============================================================================
inline void validate_bench(real_t err, benchmark::State &state) {
  if (err > 10e-6) {
    std::stringstream ss;
    ss << "Validation failed with numerical error = " << err;
    state.SkipWithError(ss.str());
  }
  if (err == 0) {
    state.SkipWithError("Validation failed with numerical error == 0. "
                        "Kernel probably didn't run");
  }
}