
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
 @file GenerateProblem.cpp

 HPCG routine
 */

#include "hpgmp.hpp"
#include "GenerateGeometry.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

#include "SetupMatrix.hpp"
#include "SetupProblem.hpp"
#include "CheckAspectRatio.hpp"
#include "OptimizeProblem.hpp"

#include "mytimer.hpp"
using std::endl;

/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESData_type, class GMRESData_type2, class Vector_type, class TestGMRESData_type>
void SetupProblem(const char *title, int argc, char ** argv, comm_type comm, int numberOfMgLevels, bool verbose,
                  Geometry * geom, SparseMatrix_type & A, GMRESData_type & data, SparseMatrix_type2 & A2, GMRESData_type2 & data2,
                  Vector_type & b, Vector_type & x, TestGMRESData_type & test_data) {

  HPCG_Params params;
  HPCG_Init_Params(title, &argc, &argv, params, comm);
  int size = params.comm_size; // Number of MPI processes
  int rank = params.comm_rank; // My process ID
  test_data.runningTime = params.runningTime;

  local_int_t nx = (local_int_t)params.nx;
  local_int_t ny = (local_int_t)params.ny;
  local_int_t nz = (local_int_t)params.nz;

  //////////////////////////////////////////////////////////
  // Construct the geometry and linear system
  GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);
  int ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank==0);


  //////////////////////////////////////////////////////////
  // Setup the problem
  bool init_vect = true;
  Vector_type xexact;
  double setup_time = mytimer();
  SetupMatrix(numberOfMgLevels, A, geom, data, &b, &x, &xexact, init_vect, comm);

  // Setup single-precision A 
  init_vect = false;
  SetupMatrix(numberOfMgLevels, A2, geom, data2, &b, &x, &xexact, init_vect, comm);
  setup_time = mytimer() - setup_time; // Capture total time of setup
  //times[9] = setup_time; // Save it for reporting
  test_data.SetupTime = setup_time;

  //////////////////////////////////////////////////////////
  // Call user-tunable set up function for A
  double opt_time = mytimer();
  OptimizeProblem(A, data, b, x, xexact);

  // Call user-tunable set up function for A2
  OptimizeProblem(A2, data, b, x, xexact);
  opt_time = mytimer() - opt_time; // Capture total time of setup
  //times[7] = opt_time;
  test_data.OptimizeTime = opt_time;

  if (verbose && A.geom->rank==0) {
    HPCG_fout << " Setup    Time     " << setup_time << " seconds." << endl;
    HPCG_fout << " Optimize Time     " << opt_time << " seconds." << endl;
  }

  //DeleteVector(xexact);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
void SetupProblem< SparseMatrix<double>, SparseMatrix<double>, GMRESData<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
 (const char*, int, char**, comm_type, int, bool, Geometry*, SparseMatrix<double>&, GMRESData<double>&, SparseMatrix<double>&, GMRESData<double>&,
  Vector<double>&, Vector<double>&, TestGMRESData<double>&);

template
void SetupProblem< SparseMatrix<float>, SparseMatrix<float>, GMRESData<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
 (const char*, int, char**, comm_type, int, bool, Geometry*, SparseMatrix<float>&, GMRESData<float>&, SparseMatrix<float>&, GMRESData<float>&,
  Vector<float>&, Vector<float>&, TestGMRESData<float>&);

// mixed
template
void SetupProblem< SparseMatrix<double>, SparseMatrix<float>, GMRESData<double>, GMRESData<float>, Vector<double>, TestGMRESData<double> >
 (const char*, int, char**, comm_type, int, bool, Geometry*, SparseMatrix<double>&, GMRESData<double>&, SparseMatrix<float>&, GMRESData<float>&,
  Vector<double>&, Vector<double>&, TestGMRESData<double>&);

#if defined(HPCG_WITH_KOKKOSKERNELS)
template
void SetupProblem< SparseMatrix<double>, SparseMatrix<half_t>, GMRESData<double>, GMRESData<half_t>, Vector<double>, TestGMRESData<double> >
 (const char*, int, char**, comm_type, int, bool, Geometry*, SparseMatrix<double>&, GMRESData<double>&, SparseMatrix<half_t>&, GMRESData<half_t>&,
  Vector<double>&, Vector<double>&, TestGMRESData<double>&);

template
void SetupProblem< SparseMatrix<double>, SparseMatrix<half_t>, GMRESData<double>, GMRESData<half_t,float>, Vector<double>, TestGMRESData<double> >
 (const char*, int, char**, comm_type, int, bool, Geometry*, SparseMatrix<double>&, GMRESData<double>&, SparseMatrix<half_t>&, GMRESData<half_t,float>&,
  Vector<double>&, Vector<double>&, TestGMRESData<double>&);
#endif
