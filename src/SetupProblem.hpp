
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

#ifndef SETUP_PROBLEM_HPP
#define SETUP_PROBLEM_HPP
#include "SetupProblem.hpp"

template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESData_type, class GMRESData_type2, class Vector_type, class TestGMRESData_type>
void SetupProblem(const char *title, int argc, char **argv, comm_type comm, int numberOfMgLevels, bool verbose, Geometry * geom,
                  SparseMatrix_type & A, GMRESData_type & data, SparseMatrix_type2 & A2, GMRESData_type2 & data2,
                  Vector_type & b, Vector_type & x, TestGMRESData_type & test_data);

#endif
