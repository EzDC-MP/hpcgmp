
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

#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"

#include "BenchGMRES.hpp"
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
int BenchGMRES(int argc, char **argv, comm_type comm, int numberOfMgLevels, bool verbose, bool runReference, TestGMRESSData_type & test_data) {

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

  Vector_type b, x, xexact;
  SetupProblem("bench_",argc, argv, comm, numberOfMgLevels, verbose, geom, A, data, A_lo, data_lo, b, x, test_data);


  // =====================================================================
  {
    local_int_t nrow = A.localNumberOfRows;
    local_int_t ncol = A.localNumberOfColumns;

    Vector_type x_overlap, b_computed;
    InitializeVector(x_overlap, ncol, A.comm);  // Overlapped copy of x vector
    InitializeVector(b_computed, nrow, A.comm); // Computed RHS vector

    // Record execution time of reference SpMV and MG kernels for reporting times
    // First load vector with random values
    FillRandomVector(x_overlap);

    int ierr = 0;
    int numberOfCalls = 10;
    double t_begin = mytimer();
    for (int i=0; i< numberOfCalls; ++i) {
      ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
      if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
      ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
      if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
    }
    //times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.
    test_data.SpmvMgTime = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.

    DeleteVector(x_overlap);
    DeleteVector(b_computed);
  }

  // =====================================================================
  // Run reference GMRES implementation for a fixed number of iterations
  // and record the obtained residual norm
  int niters = 0;
  scalar_type normr (0.0);
  scalar_type normr0 (0.0);
  int maxIters = 300;
  scalar_type tolerance = 0.0;
  int restart_length = test_data.restart_length;
  bool precond = true;
  test_data.maxNumIters = maxIters;

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

  // =====================================================================
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

  // cleanup
  DeleteMatrix(A);  
  DeleteMatrix(A_lo);
  DeleteGeometry(*geom);
  delete geom;

  DeleteGMRESData(data);
  DeleteGMRESData(data_lo);
  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform version
template
int BenchGMRES< double, double, TestGMRESData<double> >
  (int, char**, comm_type, int, bool, bool, TestGMRESData<double>&);

template
int BenchGMRES< float, float, TestGMRESData<float> >
  (int, char**, comm_type, int, bool, bool, TestGMRESData<float>&);


// mixed version
template
int BenchGMRES< double, float, TestGMRESData<double> >
  (int, char**, comm_type, int, bool, bool, TestGMRESData<double>&);

