#pragma once
#include "utils.hpp"
#include <hipSYCL/sycl/libkernel/group_functions.hpp>
#include <hipSYCL/sycl/sycl.hpp>

static constexpr auto num_error = 10e-14;

//==============================================================================
//==============================================================================
/* Solve with local accessor */
inline sycl::event solveLocal(sycl::queue &q, real_t *buffer, const Params &p) {
  auto n1 = p.n1_;
  auto n2 = p.n2_;

  const sycl::range global_size{n1, n2};
  const sycl::range local_size{1, 128};

  const sycl::nd_range ndr(global_size, local_size);

  return q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<real_t, 2> scr(local_size, cgh, sycl::no_init);

    cgh.parallel_for(ndr, [=](auto itm) {
      auto i = itm.get_global_id(0);
      auto j = itm.get_global_id(1);
      auto loci = itm.get_local_id(0);
      auto locj = itm.get_local_id(1);


      mdspan2d_t data(buffer, n1, n2);
      mdspan2d_t scratch(scr.get_pointer(), scr.get_range().get(0),
                         scr.get_range().get(1));

      scratch(loci, locj) = sycl::sin(static_cast<real_t>(i + j)) +
                            num_error; // simulating numerical error

      sycl::group_barrier(itm.get_group());

      data(i, j) = scratch(loci, locj);
    }); // end parallel_for
  });   // end q.submit
}

//==============================================================================
//==============================================================================
/* Global memory solving */
inline sycl::event solveGlobal(sycl::queue &q, real_t *buffer, real_t *scr,
                               const Params &p) {
  auto n1 = p.n1_;
  auto n2 = p.n2_;

  const sycl::range global_size{n1, n2};
  const sycl::range local_size{1, 128};

  const sycl::nd_range ndr(global_size, local_size);

  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(ndr, [=](auto itm) {
      auto i = itm.get_global_id(0);
      auto j = itm.get_global_id(1);

      mdspan2d_t data(buffer, n1, n2);
      mdspan2d_t scratch(scr, n1, n2);

      scratch(i, j) = sycl::sin(static_cast<real_t>(i + j)) +
                            num_error; // simulating numerical error

      sycl::group_barrier(itm.get_group());

      data(i, j) = scratch(i, j);
    }); // end parallel_for
  });   // end q.submit
}
