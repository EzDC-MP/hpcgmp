
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
 @file ReportResults.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <vector>
#include "ReportResults.hpp"
#include "OutputFile.hpp"
#include "OptimizeProblem.hpp"

#ifdef HPCG_DEBUG
#include <fstream>
using std::endl;

#include "hpgmp.hpp"
#endif

/*!
 Creates a YAML file and writes the information about the HPCG run, its results, and validity.

  @param[in] geom The description of the problem's geometry.
  @param[in] A    The known system matrix
  @param[in] numberOfMgLevels Number of levels in multigrid V cycle
  @param[in] niters Number of preconditioned CG iterations performed to lower the residual below a threshold
  @param[in] test_data    the data structure with the results of the CG-correctness test including pass/fail information
  @param[in] global_failure indicates whether a failure occurred during the correctness tests of CG

  @see YAML_Doc
*/
template<class SparseMatrix_type, class TestGMRESData_type>
void ReportResults(const SparseMatrix_type & A, int numberOfMgLevels,
                   const TestGMRESData_type & test_data, int global_failure) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  typedef Vector<scalar_type> Vector_type;
  typedef MGData<scalar_type> MGData_type;

  if (A.geom->rank==0) { // Only PE 0 needs to compute and report timing results

    // TODO: Put the FLOP count, Memory BW and Memory Usage models into separate functions

    // ======================== FLOP count model =======================================

    double fnrow = A.totalNumberOfRows;
    double fnnz = A.totalNumberOfNonzeros;

    // Op counts come from implementation of CG in CG.cpp (include 1 extra for the CG preamble ops)
    double fnops_ddot   = 0.0;
    double fnops_waxpby = 0.0;
    double fnops_sparsemv = 0.0;
    // Op counts from the multigrid preconditioners
    double fnops_precond = 0.0;
    double fnops = 0.0;    //fnops_ddot+fnops_waxpby+fnops_sparsemv+fnops_precond;

    // ======================== Memory bandwidth model =======================================

    // Read/Write counts come from implementation of CG in CG.cpp (include 1 extra for the CG preamble ops)
    double fnreads_ddot = 0.0;
    double fnwrites_ddot = 0.0;
    double fnreads_waxpby = 0.0;
    double fnwrites_waxpby = 0.0;
    double fnreads_sparsemv = 0.0;
    // plus nrow reads of x
    double fnwrites_sparsemv = 0.0;
    // Op counts from the multigrid preconditioners
    double fnreads_precond = 0.0;
    double fnwrites_precond = 0.0;

    const SparseMatrix_type * Af = &A;
    double fnnz_Af = Af->totalNumberOfNonzeros;
    double fnrow_Af = Af->totalNumberOfRows;
    double fnreads = fnreads_ddot+fnreads_waxpby+fnreads_sparsemv+fnreads_precond;
    double fnwrites = fnwrites_ddot+fnwrites_waxpby+fnwrites_sparsemv+fnwrites_precond;
    double frefnreads = 0.0;
    double frefnwrites = 0.0;


    // ======================== Memory usage model =======================================

    // Data in GenerateProblem_ref

    double numberOfNonzerosPerRow = 27.0; // We are approximating a 27-point finite element/volume/difference 3D stencil
    double size = ((double) A.geom->size); // Needed for estimating size of halo

    double fnbytes = ((double) sizeof(Geometry));      // Geometry struct in main.cpp

    // Model for GenerateProblem_ref.cpp
    fnbytes += fnrow*sizeof(char);      // array nonzerosInRow
    fnbytes += fnrow*((double) sizeof(global_int_t*)); // mtxIndG
    fnbytes += fnrow*((double) sizeof(local_int_t*));  // mtxIndL
    fnbytes += fnrow*((double) sizeof(double*));      // matrixValues
    fnbytes += fnrow*((double) sizeof(double*));      // matrixDiagonal
    fnbytes += fnrow*numberOfNonzerosPerRow*((double) sizeof(local_int_t));  // mtxIndL[1..nrows]
    fnbytes += fnrow*numberOfNonzerosPerRow*((double) sizeof(double));       // matrixValues[1..nrows]
    fnbytes += fnrow*numberOfNonzerosPerRow*((double) sizeof(global_int_t)); // mtxIndG[1..nrows]
    fnbytes += fnrow*((double) 3*sizeof(double)); // x, b, xexact

    // Model for GMRESData.hpp
    double fncol = ((global_int_t) A.localNumberOfColumns) * size; // Estimate of the global number of columns using the value from rank 0
    fnbytes += fnrow*((double) 2*sizeof(double)); // r, Ap
    fnbytes += fncol*((double) 2*sizeof(double)); // z, p
    // Krylov basis vectors
    int restart_length = test_data.restart_length;
    fnbytes += fnrow*((double) (restart_length+1)*sizeof(double)); // r, Ap

    std::vector<double> fnbytesPerLevel(numberOfMgLevels); // Count byte usage per level (level 0 is main CG level)
    fnbytesPerLevel[0] = fnbytes;

    // Benchmarker-provided model for OptimizeProblem.cpp
    double fnbytes_OptimizedProblem = OptimizeProblemMemoryUse(A);
    fnbytes += fnbytes_OptimizedProblem;

    Af = A.Ac;
    for (int i=1; i<numberOfMgLevels; ++i) {
      double fnrow_Af = Af->totalNumberOfRows;
      double fncol_Af = ((global_int_t) Af->localNumberOfColumns) * size; // Estimate of the global number of columns using the value from rank 0
      double fnbytes_Af = 0.0;
      // Model for GenerateCoarseProblem.cpp
      fnbytes_Af += fnrow_Af*((double) sizeof(local_int_t)); // f2cOperator
      fnbytes_Af += fnrow_Af*((double) sizeof(double)); // rc
      fnbytes_Af += 2.0*fncol_Af*((double) sizeof(double)); // xc, Axf are estimated based on the size of these arrays on rank 0
      fnbytes_Af += ((double) (sizeof(Geometry)+sizeof(SparseMatrix_type)+3*sizeof(Vector_type)+sizeof(MGData_type))); // Account for structs geomc, Ac, rc, xc, Axf - (minor)

      // Model for GenerateProblem.cpp (called within GenerateCoarseProblem.cpp)
      fnbytes_Af += fnrow_Af*sizeof(char);      // array nonzerosInRow
      fnbytes_Af += fnrow_Af*((double) sizeof(global_int_t*)); // mtxIndG
      fnbytes_Af += fnrow_Af*((double) sizeof(local_int_t*));  // mtxIndL
      fnbytes_Af += fnrow_Af*((double) sizeof(double*));      // matrixValues
      fnbytes_Af += fnrow_Af*((double) sizeof(double*));      // matrixDiagonal
      fnbytes_Af += fnrow_Af*numberOfNonzerosPerRow*((double) sizeof(local_int_t));  // mtxIndL[1..nrows]
      fnbytes_Af += fnrow_Af*numberOfNonzerosPerRow*((double) sizeof(double));       // matrixValues[1..nrows]
      fnbytes_Af += fnrow_Af*numberOfNonzerosPerRow*((double) sizeof(global_int_t)); // mtxIndG[1..nrows]

      // Model for SetupHalo_ref.cpp
#ifndef HPCG_NO_MPI
      fnbytes_Af += ((double) sizeof(double)*Af->totalToBeSent); //sendBuffer
      fnbytes_Af += ((double) sizeof(local_int_t)*Af->totalToBeSent); // elementsToSend
      fnbytes_Af += ((double) sizeof(int)*Af->numberOfSendNeighbors); // neighbors
      fnbytes_Af += ((double) sizeof(local_int_t)*Af->numberOfSendNeighbors); // receiveLength, sendLength
#endif
      fnbytesPerLevel[i] = fnbytes_Af;
      fnbytes += fnbytes_Af; // Running sum
      Af = Af->Ac; // Go to next coarse level
    }

    assert(Af==0); // Make sure we got to the lowest grid level

    // Count number of bytes used per equation
    double fnbytesPerEquation = fnbytes/fnrow;

    // Instantiate YAML document
    OutputFile doc("HPGMP-Benchmark", "1.1");
    doc.add("Release date", "March 28, 2019");

    doc.add("Machine Summary","");
    doc.get("Machine Summary")->add("Distributed Processes",A.geom->size);
    doc.get("Machine Summary")->add("Threads per processes",A.geom->numThreads);

    doc.add("Global Problem Dimensions","");
    doc.get("Global Problem Dimensions")->add("Global nx",A.geom->gnx);
    doc.get("Global Problem Dimensions")->add("Global ny",A.geom->gny);
    doc.get("Global Problem Dimensions")->add("Global nz",A.geom->gnz);

    doc.add("Processor Dimensions","");
    doc.get("Processor Dimensions")->add("npx",A.geom->npx);
    doc.get("Processor Dimensions")->add("npy",A.geom->npy);
    doc.get("Processor Dimensions")->add("npz",A.geom->npz);

    doc.add("Local Domain Dimensions","");
    doc.get("Local Domain Dimensions")->add("nx",A.geom->nx);
    doc.get("Local Domain Dimensions")->add("ny",A.geom->ny);

    int ipartz_ids = 0;
    for (int i=0; i< A.geom->npartz; ++i) {
      doc.get("Local Domain Dimensions")->add("Lower ipz", ipartz_ids);
      doc.get("Local Domain Dimensions")->add("Upper ipz", A.geom->partz_ids[i]-1);
      doc.get("Local Domain Dimensions")->add("nz",A.geom->partz_nz[i]);
      ipartz_ids = A.geom->partz_ids[i];
    }


    doc.add("########## Problem Summary  ##########","");

    doc.add("Setup Information","");
    doc.get("Setup Information")->add("Setup Time",test_data.SetupTime);

    doc.add("Linear System Information","");
    doc.get("Linear System Information")->add("Number of Equations",A.totalNumberOfRows);
    doc.get("Linear System Information")->add("Number of Nonzero Terms",A.totalNumberOfNonzeros);

    doc.add("Multigrid Information","");
    doc.get("Multigrid Information")->add("Number of coarse grid levels", numberOfMgLevels-1);
    Af = &A;
    doc.get("Multigrid Information")->add("Coarse Grids","");
    for (int i=1; i<numberOfMgLevels; ++i) {
      doc.get("Multigrid Information")->get("Coarse Grids")->add("Grid Level",i);
      doc.get("Multigrid Information")->get("Coarse Grids")->add("Number of Equations",Af->Ac->totalNumberOfRows);
      doc.get("Multigrid Information")->get("Coarse Grids")->add("Number of Nonzero Terms",Af->Ac->totalNumberOfNonzeros);
      doc.get("Multigrid Information")->get("Coarse Grids")->add("Number of Presmoother Steps",Af->mgData->numberOfPresmootherSteps);
      doc.get("Multigrid Information")->get("Coarse Grids")->add("Number of Postsmoother Steps",Af->mgData->numberOfPostsmootherSteps);
      Af = Af->Ac;
    }

    doc.add("########## Memory Use Summary  ##########","");

    doc.add("Memory Use Information","");
    doc.get("Memory Use Information")->add("Total memory used for data (Gbytes)",fnbytes/1000000000.0);
    doc.get("Memory Use Information")->add("Memory used for OptimizeProblem data (Gbytes)",fnbytes_OptimizedProblem/1000000000.0);
    doc.get("Memory Use Information")->add("Bytes per equation (Total memory / Number of Equations)",fnbytesPerEquation);

    doc.get("Memory Use Information")->add("Memory used for linear system and CG (Gbytes)",fnbytesPerLevel[0]/1000000000.0);

    doc.get("Memory Use Information")->add("Coarse Grids","");
    for (int i=1; i<numberOfMgLevels; ++i) {
      doc.get("Memory Use Information")->get("Coarse Grids")->add("Grid Level",i);
      doc.get("Memory Use Information")->get("Coarse Grids")->add("Memory used",fnbytesPerLevel[i]/1000000000.0);
    }

    /*const char DepartureFromSymmetry[] = "Departure from Symmetry |x'Ay-y'Ax|/(2*||x||*||A||*||y||)/epsilon";
    doc.add(DepartureFromSymmetry,"");
    if (testsymmetry_data.count_fail==0)
      doc.get(DepartureFromSymmetry)->add("Result", "PASSED");
    else
      doc.get(DepartureFromSymmetry)->add("Result", "FAILED");
    doc.get(DepartureFromSymmetry)->add("Departure for SpMV", testsymmetry_data.depsym_spmv);
    doc.get(DepartureFromSymmetry)->add("Departure for MG", testsymmetry_data.depsym_mg);*/

    doc.add("########## Iterations Summary  ##########","");
    doc.add("Iteration Count Information","");
    doc.get("Iteration Count Information")->add("Number of processes (validation)", test_data.validation_nprocs);
    doc.get("Iteration Count Information")->add("Restart length (validation)", test_data.restart_length);
    doc.get("Iteration Count Information")->add("Convergence tolerance (validation)", test_data.tolerance);
    doc.get("Iteration Count Information")->add("Number of reference iterations (validation)", test_data.refNumIters);
    doc.get("Iteration Count Information")->add("Initial residual norm of reference iterations (validation)", test_data.refResNorm0);
    doc.get("Iteration Count Information")->add("Final residual norm of reference iterations (validation)", test_data.refResNorm);
    doc.get("Iteration Count Information")->add("Number of optimized iterations (validation)", test_data.optNumIters);
    doc.get("Iteration Count Information")->add("Initial residual norm of optimized iterations (validation)", test_data.optResNorm0);
    doc.get("Iteration Count Information")->add("Final residual norm of optimized iterations (validation)", test_data.optResNorm);

    doc.add("########## Performance Summary (times in sec) ##########","");

    doc.add("Benchmark Time Summary","");
    doc.get("Benchmark Time Summary")->add("Run time requested (benchmark)",test_data.runningTime);
    doc.get("Benchmark Time Summary")->add("Min run time required (benchmark)",test_data.minOfficialTime);
    doc.get("Benchmark Time Summary")->add("Number of GMRES calls (benchmark)",test_data.numOfCalls);
    doc.get("Benchmark Time Summary")->add("Mximum number of iterations (benchmark)",test_data.maxNumIters);
    doc.get("Benchmark Time Summary")->add("Optimization phase",test_data.OptimizeTime);
    doc.get("Benchmark Time Summary")->add("Num SpMVs", test_data.optNumOfSPCalls);
    doc.get("Benchmark Time Summary")->add("Num MGs",   test_data.optNumOfMGCalls);
    doc.get("Benchmark Time Summary")->add("Ortho",  test_data.opt_times[3]);
    doc.get("Benchmark Time Summary")->add(" DDOT",  test_data.opt_times[1]);
    doc.get("Benchmark Time Summary")->add(" WAXPBY",test_data.opt_times[2]);
    doc.get("Benchmark Time Summary")->add("SpMV",   test_data.opt_times[4]);
    doc.get("Benchmark Time Summary")->add("MG",     test_data.opt_times[6]);
    doc.get("Benchmark Time Summary")->add("Total",  test_data.opt_times[0]);
    if (test_data.refTotalTime > 0.0) {
      doc.get("Benchmark Time Summary")->add(" - Ortho (reference)",test_data.ref_times[3]);
      doc.get("Benchmark Time Summary")->add(" - SpMV  (reference)",test_data.ref_times[4]);
      doc.get("Benchmark Time Summary")->add(" - MG    (reference)",test_data.ref_times[6]);
      doc.get("Benchmark Time Summary")->add(" - Total (reference)",test_data.refTotalTime);
    }

    doc.add("Floating Point Operations Summary","");
    doc.get("Floating Point Operations Summary")->add("Raw Ortho",test_data.opt_flops[3]);
    doc.get("Floating Point Operations Summary")->add("Raw SpMV", test_data.opt_flops[2]);
    doc.get("Floating Point Operations Summary")->add("Raw MG",   test_data.opt_flops[1]);
    doc.get("Floating Point Operations Summary")->add("Total",    test_data.opt_flops[0]);
    if (test_data.refTotalTime > 0.0) {
      doc.get("Floating Point Operations Summary")->add(" - Raw Ortho (reference)",test_data.ref_flops[3]);
      doc.get("Floating Point Operations Summary")->add(" - Raw SpMV  (reference)",test_data.ref_flops[2]);
      doc.get("Floating Point Operations Summary")->add(" - Raw MG    (reference)",test_data.ref_flops[1]);
      doc.get("Floating Point Operations Summary")->add(" - Raw Total (reference)",test_data.refTotalFlops);
    }

#if 0
    doc.add("GB/s Summary","");
    doc.get("GB/s Summary")->add("Raw Read B/W",fnreads/times[0]/1.0E9);
    doc.get("GB/s Summary")->add("Raw Write B/W",fnwrites/times[0]/1.0E9);
    doc.get("GB/s Summary")->add("Raw Total B/W",(fnreads+fnwrites)/(times[0])/1.0E9);
    doc.get("GB/s Summary")->add("Total with convergence and optimization phase overhead",(frefnreads+frefnwrites)/(times[0]+(times[7]/10.0+times[9]/10.0))/1.0E9);
#endif

    doc.add("GFLOP/s Summary","");
    doc.get("GFLOP/s Summary")->add("Raw Orho", test_data.opt_flops[3]/test_data.opt_times[3]/1.0E9);
    doc.get("GFLOP/s Summary")->add("Raw SpMV", test_data.opt_flops[2]/test_data.opt_times[4]/1.0E9);
    doc.get("GFLOP/s Summary")->add("Raw MG",   test_data.opt_flops[1]/test_data.opt_times[6]/1.0E9);
    doc.get("GFLOP/s Summary")->add("Raw Total",test_data.opt_flops[0]/test_data.opt_times[0]/1.0E9);
    if (test_data.refTotalTime > 0.0) {
      doc.get("GFLOP/s Summary")->add(" - Raw Orho  (reference)",test_data.ref_flops[3]/test_data.ref_times[3]/1.0E9);
      doc.get("GFLOP/s Summary")->add(" - Raw SpMV  (reference)",test_data.ref_flops[2]/test_data.ref_times[4]/1.0E9);
      doc.get("GFLOP/s Summary")->add(" - Raw MG    (reference)",test_data.ref_flops[1]/test_data.ref_times[6]/1.0E9);
      doc.get("GFLOP/s Summary")->add(" - Raw Total (reference)",test_data.ref_flops[0]/test_data.ref_times[0]/1.0E9);
      doc.get("GFLOP/s Summary")->add(" - Total     (reference)",test_data.refTotalFlops/test_data.refTotalTime/1.0E9);
    }
    // This final GFLOP/s rating includes the overhead of problem setup and optimizing the data structures vs ten sets of 50 iterations of CG
    double penalGflops = ((double)test_data.optNumIters) / ((double)test_data.refNumIters);
    if (penalGflops < 1.0) {
      penalGflops = 1.0;
    }
    double totalGflops = (test_data.opt_flops[0]/test_data.opt_times[0]/1.0E9) / penalGflops;
    doc.get("GFLOP/s Summary")->add("Total for benchmark",totalGflops);

    doc.add("User Optimization Overheads","");
    doc.get("User Optimization Overheads")->add("Optimization phase time (sec)", test_data.OptimizeTime);
    doc.get("User Optimization Overheads")->add("Optimization phase time vs reference SpMV+MG time", test_data.OptimizeTime/test_data.SpmvMgTime);

    doc.add("Final Summary","");
    bool isValidRun = (!global_failure);//&& (testsymmetry_data.count_fail==0) 
    if (isValidRun) {
      doc.get("Final Summary")->add("HPGMP result is VALID with a GFLOP/s rating of", totalGflops);
      if (!A.isDotProductOptimized) {
        doc.get("Final Summary")->add("Reference version of ComputeDotProduct used","Performance results are most likely suboptimal");
      }
      if (!A.isSpmvOptimized) {
        doc.get("Final Summary")->add("Reference version of ComputeSPMV used","Performance results are most likely suboptimal");
      }
      if (!A.isMgOptimized) {
        if (A.geom->numThreads>1)
          doc.get("Final Summary")->add("Reference version of ComputeMG used and number of threads greater than 1","Performance results are severely suboptimal");
        else // numThreads ==1
          doc.get("Final Summary")->add("Reference version of ComputeMG used","Performance results are most likely suboptimal");
      }
      if (!A.isWaxpbyOptimized) {
        doc.get("Final Summary")->add("Reference version of ComputeWAXPBY used","Performance results are most likely suboptimal");
      }
      if (test_data.times[0]>=test_data.minOfficialTime) {
        doc.get("Final Summary")->add("Please upload results from the YAML file contents to","http://hpcg-benchmark.org");
      }
      else {
        doc.get("Final Summary")->add("Results are valid but execution time (sec) is",test_data.times[0]);
        doc.get("Final Summary")->add("Official results execution time (sec) must be at least",test_data.minOfficialTime);
      }
    } else {
      doc.get("Final Summary")->add("HPCG result is","INVALID.");
      doc.get("Final Summary")->add("Please review the YAML file contents","You may NOT submit these results for consideration.");
    }

    std::string yaml = doc.generate();
#ifdef HPCG_DEBUG
    HPCG_fout << yaml;
#endif
  }
  return;
}


/* --------------- *
 * specializations *
 * --------------- */
template
void ReportResults< SparseMatrix<double>, TestGMRESData<double> >
  (const SparseMatrix<double>&, int, const TestGMRESData<double>&, int);

template
void ReportResults< SparseMatrix<float>, TestGMRESData<float> >
  (const SparseMatrix<float>&, int, const TestGMRESData<float>&, int);
