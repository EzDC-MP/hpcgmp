
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
 @file ComputeSPMV_ref.cpp

 HPCG routine
 */
#if defined(HPCG_WITH_SSL2)

#include "cssl.h"
#include "ComputeSPMV_ref.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include <cassert>

/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV_ref(const SparseMatrix_type & A, Vector_type & x, Vector_type & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);
  typedef typename SparseMatrix_type::scalar_type scalar_type;

  const local_int_t nrow = A.localNumberOfRows;
  scalar_type * const xv = x.values;
  scalar_type * const yv = y.values;

#ifndef HPCG_NO_MPI
  if (A.geom->size > 1) {
    ExchangeHalo(A, x);
  }
#endif

  if (std::is_same<scalar_type, double>::value) {
    int info;
    c_dvmvse ((double*)A.Ellpack_vals, nrow, 27, nrow, A.Ellpack_cols, (double*)xv, (double*)yv, &info);
    if (info != 0) printf( " Failed c_dvmvse\n" );
  } else {
    for (local_int_t i=0; i< nrow; i++)  {
      scalar_type sum = 0.0;
      const scalar_type * const cur_vals = A.matrixValues[i];
      const local_int_t * const cur_inds = A.mtxIndL[i];
      const int cur_nnz = A.nonzerosInRow[i];

      for (int j=0; j< cur_nnz; j++)
        sum += cur_vals[j]*xv[cur_inds[j]];
      yv[i] = sum;
    }
  }

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeSPMV_ref< SparseMatrix<double>, Vector<double> >(const SparseMatrix<double> &, Vector<double>&, Vector<double>&);

template
int ComputeSPMV_ref< SparseMatrix<float>, Vector<float> >(const SparseMatrix<float> &, Vector<float>&, Vector<float>&);

#endif
