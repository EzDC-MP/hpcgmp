
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

#include "BenchGMRES.hpp"
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
int BenchGMRES(SparseMatrix_type & A, SparseMatrix_type2 & A_lo, GMRESSData_type & data, GMRESSData_type2 & data_lo, Vector_type & b, Vector_type & x,
               TestGMRESSData_type & test_data, bool runReference, bool verbose) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  typedef typename SparseMatrix_type2::scalar_type scalar_type2;
  typedef Vector<scalar_type2> Vector_type2;

  // Use this array for collecting timing information
  std::vector< double > times(8,0.0);

  // Phase 1: setup
  // Run reference GMRES implementation for a fixed number of iterations
  // and record the obtained residual norm
  int niters = 0;
  scalar_type normr (0.0);
  scalar_type normr0 (0.0);
  int maxIters = 300;
  scalar_type tolerance = 0.0;
  int restart_length = test_data.restart_length;
  bool precond = true;

  int num_flops = 4;
  test_data.flops = (double*)malloc(num_flops * sizeof(double));
  for (int i=0; i<num_flops; i++) test_data.flops[i] = 0.0;
  double time_solve = 0.0;
  int numberOfGmresCalls = 10;
  if (runReference) {
    for (int i=0; i< numberOfGmresCalls; ++i) {
      ZeroVector(x); // Zero out x

      double time_tic = mytimer();
      int ierr = GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
      time_solve += (mytimer() - time_tic);

      if (ierr) HPCG_fout << "Error in call to GMRES: " << ierr << ".\n" << endl;
      if (verbose && A.geom->rank==0) {
        HPCG_fout << "Calling GMRES (all double) for testing: " << endl;
        HPCG_fout << " Number of GMRES Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        HPCG_fout << " Time     " << time_solve << " seconds." << endl;
        HPCG_fout << " (n = " << A.totalNumberOfRows << ")" << endl;
        HPCG_fout << " Time/itr " << time_solve / niters << endl;
      }
    }
    test_data.refTotalFlops = test_data.flops[0];
    test_data.refTotalTime  = time_solve;
  } else {
    test_data.refTotalFlops = 0.0;
    test_data.refTotalTime  = 0.0;
  }

  // Phase 2: benchmark
  // Run optimized GMRES (here, we are calling GMRES_IR) for a fixed number of iterations
  int num_times = 7;
  test_data.times = (double*)malloc(num_times * sizeof(double));
  for (int i=0; i<num_flops; i++) test_data.flops[i] = 0.0;
  for (int i=0; i<num_times; i++) test_data.times[i] = 0.0;
  time_solve = 0.0;
  {
    for (int i=0; i< numberOfGmresCalls; ++i) {
      ZeroVector(x); // Zero out x

      double flops = test_data.flops[0];
      double time_tic = mytimer();
      int ierr = GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
      time_solve += (mytimer() - time_tic);

      if (ierr) HPCG_fout << "Error in call to GMRES-IR: " << ierr << ".\n" << endl;
      if (verbose && A.geom->rank==0) {
        HPCG_fout << "Call [" << i << "] Number of GMRES-IR Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        HPCG_fout << " Time     " << time_solve << " seconds." << endl;
        HPCG_fout << " Gflop/s  " << flops/1000000000.0 << "/" << time_solve << " = " << (flops/1000000000.0)/time_solve 
                  << " (n = " << A.totalNumberOfRows << ")" << endl;
        HPCG_fout << " Time/itr " << time_solve / niters << endl;
      }
    }
    test_data.optTotalFlops = test_data.flops[0];
    test_data.optTotalTime = time_solve;
    test_data.numOfCalls = numberOfGmresCalls;
  }


  return 0;
}

template<class SparseMatrix_type, class GMRESSData_type, class Vector_type, class TestGMRESSData_type>
int BenchGMRES(SparseMatrix_type & A, GMRESSData_type & data, Vector_type & b, Vector_type & x, TestGMRESSData_type & test_data, bool runReference, bool verbose) {
  return BenchGMRES(A, A, data, data, b, x, test_data, runReference, verbose);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int BenchGMRES< SparseMatrix<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double>&, GMRESData<double>&, Vector<double>&, Vector<double>&, TestGMRESData<double>&, bool, bool);

template
int BenchGMRES< SparseMatrix<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
  (SparseMatrix<float>&, GMRESData<float>&, Vector<float>&, Vector<float>&, TestGMRESData<float>&, bool, bool);



// uniform version
template
int BenchGMRES< SparseMatrix<double>, SparseMatrix<double>, GMRESData<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double>&, SparseMatrix<double>&, GMRESData<double>&, GMRESData<double>&, Vector<double>&, Vector<double>&, TestGMRESData<double>&, bool, bool);

template
int BenchGMRES< SparseMatrix<float>, SparseMatrix<float>, GMRESData<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
  (SparseMatrix<float>&, SparseMatrix<float>&, GMRESData<float>&, GMRESData<float>&, Vector<float>&, Vector<float>&, TestGMRESData<float>&, bool, bool);

// mixed version
template
int BenchGMRES< SparseMatrix<double>, SparseMatrix<float>, GMRESData<double>, GMRESData<float>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double>&, SparseMatrix<float>&, GMRESData<double>&, GMRESData<float>&, Vector<double>&, Vector<double>&, TestGMRESData<double>&, bool, bool);

