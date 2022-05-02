
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
               TestGMRESSData_type & test_data, double tolerance, bool verbose) {

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
  int restart_length = 30;
  int maxIters = 300;
  // Use non-zero tolerance (one used for validation) to avoid iterating aftere convergence
  // (using tol=zero with a small number of MPI leads to GMRES continue iterating after convergence.
  //  optimized GMRES for Phase 2 converges a fewer number of iterations to converge, awarding it
  //  using extra flops from reference implementation)
  //scalar_type tolerance = 0.0;
  bool precond = true;

  int num_flops = 4;
  test_data.flops = (double*)malloc(num_flops * sizeof(double));
  for (int i=0; i<num_flops; i++) test_data.flops[i] = 0.0;
  {
    ZeroVector(x); // Zero out x

    double time_tic = mytimer();
    int ierr = GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
    double time_solve = mytimer() - time_tic;
    test_data.refTotalFlops = test_data.flops[0];

    if (ierr) HPCG_fout << "Error in call to GMRES: " << ierr << ".\n" << endl;
    //if (verbose && A.geom->rank==0) {
    if (A.geom->rank==0) {
      HPCG_fout << "Calling GMRES (all double) for testing: " << endl;
      HPCG_fout << " Number of GMRES Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
      HPCG_fout << " Time     " << time_solve << " seconds." << endl;
      HPCG_fout << " (n = " << A.totalNumberOfRows << ")" << endl;
      HPCG_fout << " Time/itr " << time_solve / niters << endl;
    }
    test_data.ref_iters = niters;
    test_data.tolerance = normr;
  }

  // Phase 2: benchmark
  // Run optimized GMRES (here, we are calling GMRES_IR), using the residual norm obtained by reference GMRES at Phase 1
  tolerance = normr;
  int numberOfGmresCalls = 10;
  test_data.normr0 = (scalar_type*)malloc(numberOfGmresCalls * sizeof(scalar_type));
  test_data.normr  = (scalar_type*)malloc(numberOfGmresCalls * sizeof(scalar_type));
  test_data.niters = (int*)malloc(numberOfGmresCalls * sizeof(int));
  test_data.niters_max = 0;
  test_data.count_pass = 0;
  test_data.count_fail = 0;

  int num_times = 7;
  test_data.times = (double*)malloc(num_times * sizeof(double));
  for (int i=0; i<num_times; i++) test_data.times[i] = 0.0;
  for (int i=0; i<num_flops; i++) test_data.flops[i] = 0.0;
  {
    for (int i=0; i< numberOfGmresCalls; ++i) {
      ZeroVector(x); // Zero out x

      double flops = test_data.flops[0];
      double time_tic = mytimer();
      int ierr = GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, precond, verbose, test_data);
      double time_solve = mytimer() - time_tic;
      flops = test_data.flops[0] - flops;

      test_data.normr0[i] = normr0;
      test_data.normr[i] = normr;
      test_data.niters[i] = niters;
      if (ierr) HPCG_fout << "Error in call to GMRES-IR: " << ierr << ".\n" << endl;
      if (normr/normr0 <= tolerance) {
        ++test_data.count_pass;
      } else {
        ++test_data.count_fail;
      }
      if (niters > test_data.niters_max) test_data.niters_max = niters;
      if (verbose && A.geom->rank==0) {
        HPCG_fout << "Call [" << i << "] Number of GMRES-IR Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        HPCG_fout << " Time     " << time_solve << " seconds." << endl;
        HPCG_fout << " Gflop/s  " << flops/1000000000.0 << "/" << time_solve << " = " << (flops/1000000000.0)/time_solve 
                  << " (n = " << A.totalNumberOfRows << ")" << endl;
        HPCG_fout << " Time/itr " << time_solve / niters << endl;
      }
    }
    test_data.numOfCalls = numberOfGmresCalls;
  }


  return 0;
}

template<class SparseMatrix_type, class GMRESSData_type, class Vector_type, class TestGMRESSData_type>
int BenchGMRES(SparseMatrix_type & A, GMRESSData_type & data, Vector_type & b, Vector_type & x, TestGMRESSData_type & test_data, double tolerance, bool verbose) {
  return BenchGMRES(A, A, data, data, b, x, test_data, tolerance, verbose);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int BenchGMRES< SparseMatrix<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double>&, GMRESData<double>&, Vector<double>&, Vector<double>&, TestGMRESData<double>&, double, bool);

template
int BenchGMRES< SparseMatrix<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
  (SparseMatrix<float>&, GMRESData<float>&, Vector<float>&, Vector<float>&, TestGMRESData<float>&, double, bool);



// uniform version
template
int BenchGMRES< SparseMatrix<double>, SparseMatrix<double>, GMRESData<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double>&, SparseMatrix<double>&, GMRESData<double>&, GMRESData<double>&, Vector<double>&, Vector<double>&, TestGMRESData<double>&, double, bool);

template
int BenchGMRES< SparseMatrix<float>, SparseMatrix<float>, GMRESData<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
  (SparseMatrix<float>&, SparseMatrix<float>&, GMRESData<float>&, GMRESData<float>&, Vector<float>&, Vector<float>&, TestGMRESData<float>&, double, bool);

// mixed version
template
int BenchGMRES< SparseMatrix<double>, SparseMatrix<float>, GMRESData<double>, GMRESData<float>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double>&, SparseMatrix<float>&, GMRESData<double>&, GMRESData<float>&, Vector<double>&, Vector<double>&, TestGMRESData<double>&, double, bool);

