
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
#if !defined(HPCG_WITH_CUDA) & !defined(HPCG_WITH_HIP)

#include "ComputeGEMV_ref.hpp"
#include "hpgmp.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMV_ref(const local_int_t m, const local_int_t n,
                    const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const SerialDenseMatrix_type & x,
                    const typename      Vector_type::scalar_type beta,  const Vector_type & y) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarX_type;
  typedef typename            Vector_type::scalar_type scalarY_type;

  const scalarA_type one  (1.0);
  const scalarA_type zero (0.0);

  assert(x.m >= n); // Test vector lengths
  assert(x.n == 1);
  assert(y.localLength >= m);

  // Input serial dense vector 
  const scalarX_type * const xv = x.values;

  scalarA_type * const Av = A.values;
  scalarY_type * const yv = y.values;

  // GEMV on HOST CPU
  if (beta == zero) {
    for (local_int_t i = 0; i < m; i++) yv[i] = zero;
  } else if (beta != one) {
    for (local_int_t i = 0; i < m; i++) yv[i] *= beta;
  }

  if (alpha == one) {
    for (local_int_t j=0; j<n; j++)
      for (local_int_t i=0; i<m; i++) {
        yv[i] += Av[i + j*m] * xv[j];
    }
  } else {
    for (local_int_t j=0; j<n; j++)
      for (local_int_t i=0; i<m; i++) {
        yv[i] += alpha * Av[i + j*m] * xv[j];
    }
  }

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMV_ref< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, SerialDenseMatrix<double> const&, double, Vector<double> const&);

template
int ComputeGEMV_ref< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, float, Vector<float> const&);


// mixed
template
int ComputeGEMV_ref< MultiVector<float>, Vector<double>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, double, Vector<double> const&);

#endif
