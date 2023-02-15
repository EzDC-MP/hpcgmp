
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
 @file ComputeSYMGS_ref.cpp

 HPCG routine
 */

#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeGS_Forward_ref.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#ifdef HPCG_DEBUG
#include "hpgmp.hpp"
#endif
#include "mytimer.hpp"
#include <cassert>
#include <iostream>

/*!

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeMG_ref(const SparseMatrix_type & A, const Vector_type & r, Vector_type & x, bool symmetric) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  #if !defined(HPCG_WITH_KOKKOSKERNELS)
  ZeroVector(x); // initialize x to zero
  #endif

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    if (symmetric) {
      for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    } else {
      for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeGS_Forward_ref(A, r, x);
    }
    if (ierr!=0) return ierr;

    // Compute residual vector
    double t0 = 0.0; TICK();
    ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;
    TOCK(x.time1);

    // Restriction operation
    TICK();
    ierr = ComputeRestriction_ref(A, r);  if (ierr!=0) return ierr;
    TOCK(x.time3);

    // MG on coarser-grid
    A.mgData->xc->time1 = A.mgData->xc->time2 = 0.0; A.mgData->xc->time3 = A.mgData->xc->time4 = 0.0;
    ierr = ComputeMG_ref(*A.Ac,*A.mgData->rc, *A.mgData->xc, symmetric);  if (ierr!=0) return ierr;
    x.time1 += A.mgData->xc->time1; x.time2 += A.mgData->xc->time2;
    x.time3 += A.mgData->xc->time3; x.time4 += A.mgData->xc->time4;

    // Prolongation operation
    TICK();
    ierr = ComputeProlongation_ref(A, x);  if (ierr!=0) return ierr;
    TOCK(x.time4);

    // Post-smoothing
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    if (symmetric) {
      for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    } else {
      for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeGS_Forward_ref(A, r, x);
    }
    if (ierr!=0) return ierr;
  }
  else {
    // coarsest grid
    if (symmetric) {
      ierr = ComputeSYMGS_ref(A, r, x);
    } else {
      ierr = ComputeGS_Forward_ref(A, r, x);
    }
    if (ierr!=0) return ierr;
  }
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeMG_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double> const&, Vector<double>&, bool);

template
int ComputeMG_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float> const&, Vector<float>&, bool);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !KOKKOS_HALF_T_IS_FLOAT // if arch does not support half, then half = float
template
int ComputeMG_ref< SparseMatrix<half_t>, Vector<half_t> >(SparseMatrix<half_t> const&, Vector<half_t> const&, Vector<half_t>&, bool);

//template
//int ComputeMG_ref< SparseMatrix<half_t>, Vector<double> >(SparseMatrix<half_t> const&, Vector<double> const&, Vector<double>&, bool);
#endif
