
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

#ifndef HPCG_NO_MPI
 #include "Utils_MPI.hpp"
#endif

#include "ComputeGEMVT_ref.hpp"
#include "hpgmp.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMVT_ref(const local_int_t m, const local_int_t n,
                     const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const Vector_type & x,
                     const typename      Vector_type::scalar_type beta,  const SerialDenseMatrix_type & y) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarX_type;
  typedef typename            Vector_type::scalar_type scalarY_type;

  const scalarA_type one  (1.0);
  const scalarA_type zero (0.0);

  assert(x.localLength >= m); // Test vector lengths
  assert(y.m >= n);
  assert(y.n == 1);

  // Input serial dense vector 
  scalarA_type * const Av = A.values;
  scalarX_type * const xv = x.values;
  scalarY_type * const yv = y.values;

  // GEMV on HOST CPU
  if (beta == zero) {
    for (local_int_t i = 0; i < n; i++) yv[i] = zero;
  } else if (beta != one) {
    for (local_int_t i = 0; i < n; i++) yv[i] *= beta;
  }

  if (alpha == one) {
    for (local_int_t j=0; j<n; j++)
      for (local_int_t i=0; i<m; i++) {
        yv[j] += Av[i + j*m] * xv[i];
    }
  } else {
    for (local_int_t i=0; i<m; i++) {
      for (local_int_t j=0; j<n; j++)
        yv[j] += alpha * Av[i + j*m] * xv[i];
    }
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarY_type>::getType ();
  MPI_Allreduce(MPI_IN_PLACE, yv, n, MPI_SCALAR_TYPE, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMVT_ref< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, Vector<double> const&, double, SerialDenseMatrix<double> const&);

template
int ComputeGEMVT_ref< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, Vector<float> const&, float, SerialDenseMatrix<float> const&);

#endif
