
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
 @file ComputeGEMMT_ref.cpp

 HPCG routine for computing GEMM transpose (dot-products)
 */
#if !defined(HPCG_WITH_CUDA) & !defined(HPCG_WITH_HIP) & !defined(HPCG_WITH_BLAS)

#ifndef HPCG_NO_MPI
 #include "Utils_MPI.hpp"
#endif

#include "ComputeGEMMT_ref.hpp"
#include "hpgmp.hpp"
#include "mytimer.hpp"

template<class MultiVector_type, class SerialDenseMatrix_type>
int ComputeGEMMT_ref(const local_int_t m, const local_int_t n, const local_int_t k,
                     const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const MultiVector_type & B,
                     const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type & C) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename            Vector_type::scalar_type scalarC_type;

  const scalarA_type one  (1.0);
  const scalarA_type zero (0.0);

  assert(x.localLength >= m); // Test vector lengths
  assert(y.m >= n);
  assert(y.n == 1);

  // Input serial dense vector 
  scalarA_type * const Av = A.values;
  scalarX_type * const Bv = B.values;
  scalarY_type * const Cv = C.values;

  // GEMM on HOST CPU
  double t0; TICK();
  if (beta == zero) {
    for (local_int_t i = 0; i < m*n; i++) Cv[i] = zero;
  } else if (beta != one) {
    for (local_int_t i = 0; i < m*n; i++) Cv[i] *= beta;
  }

  if (alpha == one) {
    for (local_int_t i=0; i<m; i++) {
      for (local_int_t j=0; j<n; j++) {
        for (local_int_t h=0; h<k; h++) {
          Cv[i + j*m] += Av[h + i*k] * Bv[j + j*k];
        }
      }
    }
  } else {
    for (local_int_t i=0; i<m; i++) {
      for (local_int_t j=0; j<n; j++) {
        for (local_int_t h=0; h<k; h++) {
          Cv[i + j*m] += alpha * Av[h + i*k] * Bv[j + j*k];
        }
      }
    }
  }
  TIME(y.time1);

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  TICK();
  int size; // Number of MPI processes, My process ID
  MPI_Comm_size(A.comm, &size);
  if (size > 1) {
    MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarY_type>::getType ();
    MPI_Allreduce(MPI_IN_PLACE, Cv, m*n, MPI_SCALAR_TYPE, MPI_SUM, A.comm);
  }
  TIME(y.time2);
#else
  y.time2 = 0.0;
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMMT_ref< MultiVector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, MultiVector<double> const&, double, SerialDenseMatrix<double> &);

template
int ComputeGEMMT_ref< MultiVector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, MultiVector<float> const&, float, SerialDenseMatrix<float> &);

#endif
