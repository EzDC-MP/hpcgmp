
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

#ifndef REPORTRESULTS_HPP
#define REPORTRESULTS_HPP
#include "SparseMatrix.hpp"
#include "TestGMRES.hpp"

template<class SparseMatrix_type, class TestGMRESData_type>
void ReportResults(const SparseMatrix_type & A, int numberOfMgLevels,
                   const TestGMRESData_type & testcg_data, int global_failure, bool quickPath);

#endif // REPORTRESULTS_HPP
