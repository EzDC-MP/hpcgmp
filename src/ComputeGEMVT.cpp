
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
 @file ComputeGEMVT.cpp

 Routine to compute the GEMV of transpose of a matrix and a vector.
 */
#include "ComputeGEMVT.hpp"
#include "ComputeGEMVT_ref.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMVT(const local_int_t m, const local_int_t n,
                 const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const Vector_type & x,
                 const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type & y,
                 bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeGEMV should be used.
  isOptimized = false;
  return ComputeGEMVT_ref(m, n, alpha, A, x, beta, y);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMVT< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, Vector<double> const&, const double, SerialDenseMatrix<double> &, bool&);

template
int ComputeGEMVT< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, Vector<float> const&, const float, SerialDenseMatrix<float> &, bool&);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !defined(KOKKOS_HALF_T_IS_FLOAT) // if arch does not support half, then half = float
template
int ComputeGEMVT< MultiVector<half_t>, Vector<half_t>, SerialDenseMatrix<half_t> >
  (int, int, half_t, MultiVector<half_t> const&, Vector<half_t> const&, const half_t, SerialDenseMatrix<half_t> &, bool&);

template
int ComputeGEMVT< MultiVector<half_t>, Vector<half_t>, SerialDenseMatrix<float> >
  (int, int, half_t, MultiVector<half_t> const&, Vector<half_t> const&, const float, SerialDenseMatrix<float> &, bool&);
#endif
