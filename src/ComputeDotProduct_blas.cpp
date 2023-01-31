
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeDotProduct_ref.cpp

 HPCG routine
 */
#if defined(HPCG_WITH_BLAS)

#include "cblas.h"
#ifndef HPCG_NO_MPI
 #include <mpi.h>
 #include "mytimer.hpp"
 #include "Utils_MPI.hpp"
#endif
#include "ComputeDotProduct_ref.hpp"
#include "hpgmp.hpp"

#ifndef HPCG_NO_OPENMP
 #include <omp.h>
#endif
#include <cassert>

/*!
  Routine to compute the dot product of two vectors where:

  This is the reference dot-product implementation.  It _CANNOT_ be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] x, y the input vectors
  @param[in] result a pointer to scalar value, on exit will contain result.
  @param[out] time_allreduce the time it took to perform the communication between processes

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
*/
template<class Vector_type>
int ComputeDotProduct_ref(const local_int_t n, const Vector_type & x, const Vector_type & y,
                          typename Vector_type::scalar_type & result, double & time_allreduce) {
  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  typedef typename Vector_type::scalar_type scalar_type;
  scalar_type local_result (0.0);

  scalar_type* xv = x.values;
  scalar_type* yv = y.values;

  // Compute dot on Nvidia GPU
  if (std::is_same<scalar_type, double>::value) {
    local_result = cblas_ddot(n, (double*)xv, 1, (double*)yv, 1);
  } else if (std::is_same<scalar_type, float>::value) {
    local_result = cblas_sdot(n, (float*)xv, 1, (float*)yv, 1);
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType ();
  double t0 = mytimer();
  scalar_type global_result (0.0);
  MPI_Allreduce(&local_result, &global_result, 1, MPI_SCALAR_TYPE, MPI_SUM, x.comm);
  result = global_result;
  time_allreduce += mytimer() - t0;

#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeDotProduct_ref<Vector<double> >(int, Vector<double> const&, Vector<double> const&, double&, double&);

template
int ComputeDotProduct_ref<Vector<float> >(int, Vector<float> const&, Vector<float> const&, float&, double&);

#endif
