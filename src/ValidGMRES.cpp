
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

#include "ValidGMRES.hpp"
#include "GMRES.hpp"
#include "GMRES_IR.hpp"
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


template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESSData_type, class GMRESSData_type2, class Vector_type, class TestGMRESSData_type>
int ValidGMRES(SparseMatrix_type & A, SparseMatrix_type2 & A_lo, GMRESSData_type & data, GMRESSData_type2 & data_lo, Vector_type & b, Vector_type & x,
               TestGMRESSData_type & test_data, bool verbose) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  typedef typename SparseMatrix_type2::scalar_type scalar_type2;
  typedef Vector<scalar_type2> Vector_type2;


  int restart_length = 30;
  int MaxIters = 3000;
  scalar_type tolerance = 1e-9;

  test_data.tolerance = tolerance;
  test_data.restart_length = restart_length;

  //if (A.geom->rank == 0 && verbose) 
  {
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
  }
  if (A.geom->rank == 0 && refResNorm/refResNorm0 > tolerance) {
    HPCG_fout << " ref GMRES did not converege: normr = " << refResNorm/refResNorm0 << "(tol = " << tolerance << ")" << endl;
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
  }
  if (A.geom->rank == 0 && optResNorm/optResNorm0 > tolerance) {
    fail = 1;
    HPCG_fout << " opt GMRES did not converege: normr = " << optResNorm/optResNorm0 << "(tol = " << tolerance << ")" << endl;
  }

  return fail;
}

template<class SparseMatrix_type, class GMRESSData_type, class Vector_type, class TestGMRESSData_type>
int ValidGMRES(SparseMatrix_type & A, GMRESSData_type & data, Vector_type & b, Vector_type & x, TestGMRESSData_type & test_data, bool verbose) {
  return ValidGMRES(A, A, data, data, b, x, test_data, verbose);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ValidGMRES< SparseMatrix<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double>&, GMRESData<double>&, Vector<double>&, Vector<double>&, TestGMRESData<double>&, bool);

template
int ValidGMRES< SparseMatrix<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
  (SparseMatrix<float>&, GMRESData<float>&, Vector<float>&, Vector<float>&, TestGMRESData<float>&, bool);



// uniform version
template
int ValidGMRES< SparseMatrix<double>, SparseMatrix<double>, GMRESData<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double>&, SparseMatrix<double>&, GMRESData<double>&, GMRESData<double>&, Vector<double>&, Vector<double>&, TestGMRESData<double>&, bool);

template
int ValidGMRES< SparseMatrix<float>, SparseMatrix<float>, GMRESData<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
  (SparseMatrix<float>&, SparseMatrix<float>&, GMRESData<float>&, GMRESData<float>&, Vector<float>&, Vector<float>&, TestGMRESData<float>&, bool);

// mixed version
template
int ValidGMRES< SparseMatrix<double>, SparseMatrix<float>, GMRESData<double>, GMRESData<float>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double>&, SparseMatrix<float>&, GMRESData<double>&, GMRESData<float>&, Vector<double>&, Vector<double>&, TestGMRESData<double>&, bool);

