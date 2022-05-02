
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
 @file SetupProblem.cpp

 HPCG routine
 */

#ifndef SETUP_PROBLEM_HPP
#define SETUP_PROBLEM_HPP

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "GMRESData.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GenerateNonsymProblem.hpp"
#include "GenerateNonsymCoarseProblem.hpp"
#include "SetupHalo.hpp"



/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

template<class SparseMatrix_type, class GMRESData_type, class Vector_type>
void SetupProblem(int numberOfMgLevels, SparseMatrix_type & A, Geometry * geom, GMRESData_type & data, Vector_type * b, Vector_type * x, Vector_type * xexact, bool init_vect);

#endif
