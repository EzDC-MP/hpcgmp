
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
#if 0 //defined(HPCG_WITH_KOKKOSKERNELS)

#include "ComputeSPMV_ref.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
 #include <omp.h>
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

#if 1
  #ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
  #endif
  for (local_int_t i=0; i< nrow; i++)  {
    scalar_type sum = 0.0;
    const scalar_type * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
#else
  {
    const int nnzA = A.localNumberOfNonzeros;
    typename SparseMatrix_type::RowPtrView rowptr_view(A.h_row_ptr, nrow+1);
    typename SparseMatrix_type::ColIndView colidx_view(A.h_col_idx, nnzA);
    typename SparseMatrix_type::ValuesView values_view(A.h_nzvals,  nnzA);
    graph_t static_graph(column_view, rowmap_view);
    crsmat_t crsmat("CrsMatrix", n, values_view, static_graph);

    typename SparseMatrix_type::ValuesView x_view(x.values, ncol);
    typename SparseMatrix_type::ValuesView y_view(y.values, nrow);
    KokkosSparse::spmv(tran, one, A_view, x_view, one, y_view);
  }
#endif
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeSPMV_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double>&, Vector<double>&);

template
int ComputeSPMV_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float>&, Vector<float>&);

#endif
