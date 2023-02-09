
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
#if defined(HPCG_WITH_KOKKOSKERNELS)

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
  const local_int_t ncol = A.localNumberOfColumns;
  scalar_type * const xv = x.values;
  scalar_type * const yv = y.values;

#ifndef HPCG_NO_MPI
  if (A.geom->size > 1) {
    #ifdef HPCG_WITH_CUDA
    // Copy local part of X to HOST CPU
    if (cudaSuccess != cudaMemcpy(xv, x.d_values, nrow*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_y\n" );
    }
    #elif defined(HPCG_WITH_HIP)
    if (hipSuccess != hipMemcpy(xv, x.d_values, nrow*sizeof(scalar_type), hipMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_y\n" );
    }
    #endif

    ExchangeHalo(A, x);

    // copy non-local part of X to device (after Halo exchange)
    #if defined(HPCG_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(&x.d_values[nrow], &xv[nrow], (ncol-nrow)*sizeof(scalar_type), cudaMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_x\n" );
    }
    #elif defined(HPCG_WITH_HIP)
    if (hipSuccess != hipMemcpy(&x.d_values[nrow], &xv[nrow], (ncol-nrow)*sizeof(scalar_type), hipMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_x\n" );
    }
    #endif
  }
#endif

#if 0
  // Copy input vectors from HOST CPU
  #if defined(HPCG_WITH_CUDA)
  scalar_type * const d_xv = x.d_values;
  if (cudaSuccess != cudaMemcpy(xv, d_xv, ncol*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_x\n" );
  }
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

  // Copy output vector Y from HOST CPU
  #if defined(HPCG_WITH_CUDA)
  if (cudaSuccess != cudaMemcpy(d_xv, xv, nrow*sizeof(scalar_type), cudaMemcpyHostToDevice)) {
    printf( " Failed to memcpy d_x\n" );
  }
  #endif
#else
  {
    const int nnzA = A.localNumberOfNonzeros;
    #if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
    typename SparseMatrix_type::RowPtrView rowptr_view(A.d_row_ptr, nrow+1);
    typename SparseMatrix_type::ColIndView colidx_view(A.d_col_idx, nnzA);
    typename SparseMatrix_type::ValuesView values_view(A.d_nzvals,  nnzA);

    typename SparseMatrix_type::ValuesView x_view(x.d_values, ncol);
    typename SparseMatrix_type::ValuesView y_view(y.d_values, nrow);
    #else
    typename SparseMatrix_type::RowPtrView rowptr_view(A.h_row_ptr, nrow+1);
    typename SparseMatrix_type::ColIndView colidx_view(A.h_col_idx, nnzA);
    typename SparseMatrix_type::ValuesView values_view(A.h_nzvals,  nnzA);

    typename SparseMatrix_type::ValuesView x_view(x.values, ncol);
    typename SparseMatrix_type::ValuesView y_view(y.values, nrow);
    #endif
    typename SparseMatrix_type::StaticGraphView static_graph(colidx_view, rowptr_view);
    typename SparseMatrix_type::CrsMatView A_view("CrsMatrix", ncol, values_view, static_graph);

    const scalar_type one  (1.0);
    const scalar_type zero (0.0);
    KokkosSparse::spmv(KokkosSparse::NoTranspose, one, A_view, x_view, zero, y_view);
  }
#endif
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeSPMV_ref< SparseMatrix<double>, Vector<double> >(const SparseMatrix<double> &, Vector<double>&, Vector<double>&);

template
int ComputeSPMV_ref< SparseMatrix<float>, Vector<float> >(const SparseMatrix<float> &, Vector<float>&, Vector<float>&);

template
int ComputeSPMV_ref< SparseMatrix<half_t>, Vector<half_t> >(const SparseMatrix<half_t> &, Vector<half_t>&, Vector<half_t>&);
#endif
