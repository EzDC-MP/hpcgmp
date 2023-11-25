
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
 @file ComputeGEMMT.cpp

 Routine to compute the GEMM of transpose of a matrix and a vector.
 */
#include "ComputeGEMMT.hpp"
#include "ComputeGEMMT_ref.hpp"

template<class MultiVector_type, class SerialDenseMatrix_type>
int ComputeGEMMT(const local_int_t m, const local_int_t n, const local_int_t k,
                 const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const MultiVector_type & B,
                 const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type & C,
                 bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeGEMM should be used.
  isOptimized = false;
  return ComputeGEMMT_ref(m, n, k, alpha, A, B, beta, C);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMMT< MultiVector<double>, SerialDenseMatrix<double> >
  (const int, const int, const int, const double, MultiVector<double> const&, MultiVector<double> const&, const double, SerialDenseMatrix<double> &, bool&);

template
int ComputeGEMMT< MultiVector<float>, SerialDenseMatrix<float> >
  (const int, const int, const int, const float, MultiVector<float> const&, MultiVector<float> const&, const float, SerialDenseMatrix<float> &, bool&);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !defined(KOKKOS_HALF_T_IS_FLOAT) // if arch does not support half, then half = float
template
int ComputeGEMMT< MultiVector<half_t>, SerialDenseMatrix<half_t> >
  (int, int, int, half_t, MultiVector<half_t> const&, MultiVector<half_t> const&, const half_t, SerialDenseMatrix<half_t> &, bool&);

template
int ComputeGEMMT< MultiVector<half_t>, SerialDenseMatrix<float> >
  (int, int, int, half_t, MultiVector<half_t> const&, MultiVector<half_t> const&, const float, SerialDenseMatrix<float> &, bool&);

template
int ComputeGEMMT< MultiVector<half_t>, SerialDenseMatrix<double> >
  (int, int, int, half_t, MultiVector<half_t> const&, MultiVector<half_t> const&, const double, SerialDenseMatrix<double> &, bool&);
#endif
