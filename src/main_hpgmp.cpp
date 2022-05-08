
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
 @file main.cpp

 HPGMP routine
 */

// Main routine of a program that calls the HPGMP GMRES and GMRES-IR 
// solvers to solve the problem, and then prints results.

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
using std::endl;

#include <vector>

#include "hpgmp.hpp"

#include "SetupProblem.hpp"
#include "CheckAspectRatio.hpp"
#include "GenerateGeometry.hpp"
#include "SetupHalo.hpp"
#include "CheckProblem.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"

#include "GMRESData.hpp"
#include "ValidGMRES.hpp"
#include "BenchGMRES.hpp"

typedef double scalar_type;
typedef TestGMRESData<scalar_type> TestGMRESData_type;

typedef Vector<scalar_type> Vector_type;
typedef SparseMatrix<scalar_type> SparseMatrix_type;
typedef GMRESData<scalar_type> GMRESData_type;

typedef float scalar_type2;
typedef Vector<scalar_type2> Vector_type2;
typedef SparseMatrix<scalar_type2> SparseMatrix_type2;
typedef GMRESData<scalar_type2> GMRESData_type2;


/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char * argv[]) {

#ifndef HPCG_NO_MPI
  MPI_Init(&argc, &argv);
#endif
  int numRanks;
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  //////////////////////////
  // Create Communicators //
  //////////////////////////
  MPI_Comm valid_comm = MPI_COMM_WORLD;

  int color = 0;
  int sizeValidComm = 4;
  if (sizeValidComm > numRanks) {
    #if 1
    sizeValidComm = numRanks;
    #else
    asseert(1);
    #endif
  }

  if (myRank < sizeValidComm) {
    color = 1;
  }
  MPI_Comm_split(MPI_COMM_WORLD, color, myRank, &valid_comm);

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  bool quickPath = 1; //TODO: Change back to the following after=(params.runningTime==0);
  int numberOfMgLevels = 4; // Number of levels including first

  HPCG_Params params;


  // Use this array for collecting timing information
  bool verbose = false;
  std::vector< double > times(10,0.0);

  int ierr = 0;  // Used to check return codes on function calls
  TestGMRESData_type test_data;
  test_data.times = NULL;
  test_data.flops = NULL;


  /////////////////////////
  // Problem setup Phase //
  /////////////////////////
  Geometry * geom = new Geometry;

  SparseMatrix_type A;
  GMRESData_type data;

  SparseMatrix_type2 A2;
  GMRESData_type2 data2;

  Vector_type b, x, xexact;
  SetupProblem(argc, argv, MPI_COMM_WORLD, numberOfMgLevels, verbose, geom, A, data, A2, data2, b, x, test_data);

  //////////////////////////////////////////////////////////////////////////
  // Validation Phase: make sure optimized version converges to specified //
  //////////////////////////////////////////////////////////////////////////

  //////////////////////
  // Validation phase //
  //////////////////////
  int global_failure = 0;
  ValidGMRES(A, A2, data, data2, b, x, test_data, verbose);


  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////
  // Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines
  MPI_Comm bench_comm = MPI_COMM_WORLD;

  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector_type x_overlap, b_computed;
  InitializeVector(x_overlap, ncol, bench_comm);  // Overlapped copy of x vector
  InitializeVector(b_computed, nrow, bench_comm); // Computed RHS vector


  // Record execution time of reference SpMV and MG kernels for reporting times
  // First load vector with random values
  FillRandomVector(x_overlap);

  int numberOfCalls = 10;
  if (quickPath) numberOfCalls = 1; //QuickPath means we do on one call of each block of repetitive code
  double t_begin = mytimer();
  for (int i=0; i< numberOfCalls; ++i) {
    ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
    if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
    if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  }
  times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.


  /////////////////////
  // Benchmark phase //
  /////////////////////
  bool runReference = true;
  BenchGMRES(A, A2, data, data2, b, x, test_data, runReference, verbose);

  
  ////////////////////
  // Report Results //
  ////////////////////

  // Report results to YAML file
  ReportResults(A, numberOfMgLevels, test_data, global_failure, quickPath);

  // Clean up
  DeleteMatrix(A); // This delete will recursively delete all coarse grid data
  DeleteGMRESData(data);
  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);
  DeleteVector(x_overlap);
  DeleteVector(b_computed);

  // Finish up
  HPCG_Finalize();
#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
