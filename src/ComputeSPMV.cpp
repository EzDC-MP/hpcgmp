
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV(const SparseMatrix_type & A, Vector_type & x, Vector_type & y) {

  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  A.isSpmvOptimized = false;
  return ComputeSPMV_ref(A, x, y);
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeSPMV< SparseMatrix<double>, Vector<double> >(const SparseMatrix<double> &, Vector<double>&, Vector<double>&);

template
int ComputeSPMV< SparseMatrix<float>, Vector<float> >(const SparseMatrix<float> &, Vector<float>&, Vector<float>&);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !KOKKOS_HALF_T_IS_FLOAT // if arch does not support half, then half = float
template
int ComputeSPMV< SparseMatrix<half_t>, Vector<half_t> >(const SparseMatrix<half_t> &, Vector<half_t>&, Vector<half_t>&);
#endif
