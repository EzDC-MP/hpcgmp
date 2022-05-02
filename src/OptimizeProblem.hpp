
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

#ifndef OPTIMIZEPROBLEM_HPP
#define OPTIMIZEPROBLEM_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"

template<class SparseMatrix_type, class GMRESData_type, class Vector_type>
int OptimizeProblem(SparseMatrix_type & A, GMRESData_type & data, Vector_type & b, Vector_type & x, Vector_type & xexact);

// This helper function should be implemented in a non-trivial way if OptimizeProblem is non-trivial
// It should return as type double, the total number of bytes allocated and retained after calling OptimizeProblem.
// This value will be used to report Gbytes used in ReportResults (the value returned will be divided by 1000000000.0).

template<class SparseMatrix_type>
double OptimizeProblemMemoryUse(const SparseMatrix_type & A);

#endif  // OPTIMIZEPROBLEM_HPP
