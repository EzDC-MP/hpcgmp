
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
 @file Vector.hpp

 HPCG data structures for dense vectors
 */

#include "ComputeTRSM.hpp"

template<class SerialDenseMatrix_type>
int ComputeTRSM(const local_int_t n,
                const typename SerialDenseMatrix_type::scalar_type alpha,
                const SerialDenseMatrix_type & U,
                      SerialDenseMatrix_type & x) {

  typedef typename SerialDenseMatrix_type::scalar_type scalar_type;
  const scalar_type one  (1.0);
  const scalar_type zero (0.0);

  assert(x.m >= n);
  assert(x.n == 1); // one RHS

  const local_int_t m = U.m;
  const scalar_type * const Uv = U.values;
  scalar_type * xv = x.values;

  for (local_int_t i = n-1; i >= 0; i--) {
    for (local_int_t j = i+1; j < n; j++)
      xv[i] -= Uv[i + j*m] * xv[j];
    xv[i] /= Uv[i + i*m];
  }
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeTRSM< SerialDenseMatrix<double> >(int, double, SerialDenseMatrix<double> const&, SerialDenseMatrix<double>&);

template
int ComputeTRSM< SerialDenseMatrix<float> >(int, float, SerialDenseMatrix<float> const&, SerialDenseMatrix<float>&);
