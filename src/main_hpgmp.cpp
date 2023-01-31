
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

#ifdef HPCG_WITH_KOKKOSKERNELS
#include "Kokkos_Core.hpp"
#endif

#include "hpgmp.hpp"

#include "SetupProblem.hpp"
#include "ReportResults.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"

#include "GMRESData.hpp"
#include "ValidGMRES.hpp"
#include "BenchGMRES.hpp"
#include "mytimer.hpp"

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
#ifdef HPCG_WITH_KOKKOSKERNELS
  Kokkos::initialize();
  {
#endif
  int numRanks;
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  //////////////////////////
  // Create Communicators //
  //////////////////////////
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
  MPI_Comm validation_comm = MPI_COMM_WORLD;
  MPI_Comm_split(MPI_COMM_WORLD, color, myRank, &validation_comm);

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  int numberOfMgLevels = 4; // Number of levels including first


  // Use this array for collecting timing information
  bool verbose = false;
  TestGMRESData_type test_data;
  test_data.times = NULL;
  test_data.flops = NULL;
  test_data.validation_nprocs = sizeValidComm;


  //////////////////////
  // Validation phase //
  //////////////////////
  int global_failure = 0;
  int restart_length = 30;
  scalar_type tolerance = 1e-9;

  test_data.tolerance = tolerance;
  test_data.restart_length = restart_length;
  if (myRank < sizeValidComm) {
    global_failure = ValidGMRES<scalar_type, scalar_type2> (argc, argv, validation_comm, numberOfMgLevels, verbose, test_data);
    HPCG_Finalize();
  }


  /////////////////////
  // Benchmark phase //
  /////////////////////
  {
    bool runReference = true;
    BenchGMRES<scalar_type, scalar_type2>(argc, argv, MPI_COMM_WORLD, numberOfMgLevels, verbose, runReference, test_data);
    MPI_Barrier(MPI_COMM_WORLD);
    HPCG_Finalize();
  }

  
  ////////////////////
  // Report Results //
  ////////////////////
  {
    // setup problem for reporting (TODO: remove)
    Geometry * geom = new Geometry;

    SparseMatrix_type A;
    GMRESData_type data;

    SparseMatrix_type2 A2;
    GMRESData_type2 data2;

    Vector_type b, x;
    SetupProblem("report_", argc, argv, MPI_COMM_WORLD, numberOfMgLevels, verbose, geom, A, data, A2, data2, b, x, test_data);


    // Report results to YAML file
    ReportResults(A, numberOfMgLevels, test_data, global_failure);

    // Clean up
    DeleteMatrix(A);
    DeleteMatrix(A2);
    DeleteGeometry(*geom);
    delete geom;

    DeleteGMRESData(data);
    DeleteGMRESData(data2);
    DeleteVector(x);
    DeleteVector(b);
  }

  // Finish up
  HPCG_Finalize();
#ifdef HPCG_WITH_KOKKOSKERNELS
  }
  Kokkos::finalize();
#endif
#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
