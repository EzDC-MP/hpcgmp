
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
 @file TestGMRES.cpp

 HPCG routine
 */

// Changelog
//
// Version 0.4
// - Added timing of setup time for sparse MV
// - Corrected percentages reported for sparse MV with overhead
//
/////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
using std::endl;
#include <vector>
#include "hpgmp.hpp"

#include "SetupProblem.hpp"
#include "GMRES.hpp"
#include "GMRES_IR.hpp"

#include "ValidGMRES.hpp"
#include "mytimer.hpp"

/*!
  Test the correctness of the Preconditined CG implementation by using a system matrix with a dominant diagonal.

  @param[in]    geom The description of the problem's geometry.
  @param[in]    A    The known system matrix
  @param[in]    data the data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[out]   test_data the data structure with the results of the test including pass/fail information

  @return Returns zero on success and a non-zero value otherwise.

  @see CG()
 */


template<class scalar_type, class scalar_type2, class TestGMRESSData_type>
int ValidGMRES(int argc, char **argv, comm_type comm, int numberOfMgLevels, bool verbose, TestGMRESSData_type & test_data) {

  typedef Vector<scalar_type> Vector_type;
  typedef SparseMatrix<scalar_type> SparseMatrix_type;
  typedef GMRESData<scalar_type> GMRESData_type;

  typedef Vector<scalar_type2> Vector_type2;
  typedef SparseMatrix<scalar_type2> SparseMatrix_type2;
  typedef GMRESData<scalar_type2> GMRESData_type2;


  //////////////////////////////////////////////////////////
  // Setup problem
  Geometry * geom = new Geometry;

  SparseMatrix_type A;
  GMRESData_type data;

  SparseMatrix_type2 A_lo;
  GMRESData_type2 data_lo;

  Vector_type b, x;
  SetupProblem("valid_", argc, argv, comm, numberOfMgLevels, verbose, geom, A, data, A_lo, data_lo, b, x, test_data);


  //////////////////////////////////////////////////////////
  // Solver Parameters
  int MaxIters = 3000;
  int restart_length = test_data.restart_length;
  scalar_type tolerance = test_data.tolerance;
  if (A.geom->rank == 0 && verbose) {
    HPCG_fout << endl << " >> In Validate GMRES( tol = " << tolerance << " and restart = " << restart_length << ") <<" << endl;
  }


  //////////////////////////////////////////////////////////
  // Run reference GMRES to a fixed tolerance
  int refNumIters = 0;
  scalar_type refResNorm = 0.0;
  scalar_type refResNorm0 = 0.0;
  {
    ZeroVector(x);
    int ierr = GMRES(A, data, b, x, restart_length, MaxIters, tolerance, refNumIters, refResNorm, refResNorm0, true, verbose, test_data);
    test_data.refNumIters = refNumIters;
    test_data.refResNorm0 = refResNorm0;
    test_data.refResNorm  = refResNorm;
  }
  if (A.geom->rank == 0 && refResNorm/refResNorm0 > tolerance) {
    HPCG_fout << " ref GMRES did not converege: normr = " << refResNorm << " / " << refResNorm0 << " = " << refResNorm/refResNorm0 << "(tol = " << tolerance << ")" << endl;
  }


  //////////////////////////////////////////////////////////
  // Run "optimized" GMRES (aka GMRES-IR) to a fixed tolerance
  int fail = 0;
  int optNumIters = 0;
  scalar_type optResNorm = 0.0;
  scalar_type optResNorm0 = 0.0;
  {
    ZeroVector(x);
    int ierr = GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, MaxIters, tolerance, optNumIters, optResNorm, optResNorm0, true, verbose, test_data);
    test_data.optNumIters = optNumIters;
    test_data.optResNorm0 = optResNorm0;
    test_data.optResNorm  = optResNorm;
  }
  if (optResNorm/optResNorm0 > tolerance) {
    fail = 1;
    if (A.geom->rank == 0) {
      HPCG_fout << " opt GMRES did not converege: normr = " << optResNorm << " / " << optResNorm0 << " = " << optResNorm/optResNorm0 << "(tol = " << tolerance << ")" << endl;
    }
  }


  // cleanup
  DeleteMatrix(A);
  DeleteMatrix(A_lo);
  DeleteGeometry(*geom);
  delete geom;

  DeleteGMRESData(data);
  DeleteGMRESData(data_lo);
  DeleteVector(x);
  DeleteVector(b);

  return fail;
}



/* --------------- *
 * specializations *
 * --------------- */

// uniform version
template
int ValidGMRES< double, double, TestGMRESData<double> > (int, char**, comm_type, int, bool, TestGMRESData<double>&);

template
int ValidGMRES< float, float, TestGMRESData<float> > (int, char**, comm_type, int, bool, TestGMRESData<float>&);

// mixed version
template
int ValidGMRES< double, float, TestGMRESData<double> > (int, char**, comm_type, int, bool, TestGMRESData<double>&);

