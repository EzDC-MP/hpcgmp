
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
 @file ComputeGS_Forward_ref.cpp

 HPCG routine
 */
#if defined(HPCG_WITH_KOKKOSKERNELS)

#ifndef HPCG_NO_MPI
 #include "ExchangeHalo.hpp"
#endif
#include "ComputeGS_Forward_ref.hpp"
#include <cassert>
#include <iostream>

#include "ComputeSPMV.hpp"
#include "ComputeWAXPBY.hpp"
#ifdef HPCG_DEBUG
 #include <mpi.h>
 #include "Utils_MPI.hpp"
 #include "hpgmp.hpp"
#endif

/*!
  Computes one forward step of Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeGS_Forward
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeGS_Forward_ref(const SparseMatrix_type & A, const Vector_type & r, Vector_type & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  const local_int_t nrow = A.localNumberOfRows;
  const local_int_t ncol = A.localNumberOfColumns;

#ifndef HPCG_NO_MPI
  // Exchange Halo on HOST CPU
  ExchangeHalo(A, x);
  #ifdef HPCG_DEBUG
  if (A.geom->rank==0) {
    HPCG_fout << A.geom->rank << " : ComputeGS(" << nrow << " x " << ncol << ") start" << std::endl;
  }
  #endif
#endif

#if 1
  const scalar_type * const rv = r.values;
  scalar_type * const xv = x.values;
  scalar_type ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues

  for (local_int_t i=0; i < nrow; i++) {
    const scalar_type * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const scalar_type currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    scalar_type sum = rv[i]; // RHS value

    for (int j=0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
  }

  return 0;
#else
  {
    // wrap pointers into Kokkos Views and call, Kokkos-Kernels forward-sweep Gauss-Seidel
    bool init_zero_x_vector = true;
    bool update_y_vector = true;
    const scalar_type omega (1.0);
    int num_sweeps = 1;

    const int nnzA = A.localNumberOfNonzeros;
    typename SparseMatrix_type::RowPtrView rowptr_view(A.h_row_ptr, nrow+1);
    typename SparseMatrix_type::ColIndView colidx_view(A.h_col_idx, nnzA);
    typename SparseMatrix_type::ValuesView values_view(A.h_nzvals,  nnzA);

    typename SparseMatrix_type::ValuesView r_view(r.values, ncol);
    typename SparseMatrix_type::ValuesView x_view(x.values, nrow);
    typename SparseMatrix_type::KernelHandle *handle = const_cast<typename SparseMatrix_type::KernelHandle*>(&(A.kh));
    KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply
      (handle, nrow, ncol, rowptr_view, colidx_view, values_view, x_view, r_view, init_zero_x_vector, update_y_vector, omega, num_sweeps);
    return 0;
  }
#endif
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeGS_Forward_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double> const&, Vector<double>&);

template
int ComputeGS_Forward_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float> const&, Vector<float>&);

#endif

