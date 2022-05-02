
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
 @file BenchGMRES.hpp

 HPGMRES data structure
 */

#ifndef BENCHGMRES_HPP
#define BENCHGMRES_HPP

#include "hpgmp.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"

template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESData_type, class GMRESData_type2, class Vector_type, class TestGMRESData_type>
extern int BenchGMRES(SparseMatrix_type & A, SparseMatrix_type2 & A_lo, GMRESData_type & data, GMRESData_type2 & data_lo, Vector_type & b, Vector_type & x, TestGMRESData_type & testcg_data);

template<class SparseMatrix_type, class GMRESData_type, class Vector_type, class TestGMRESData_type>
extern int BenchGMRES(SparseMatrix_type & A, GMRESData_type & data, Vector_type & b, Vector_type & x, TestGMRESData_type & testcg_data);

#endif  // BENCHGMRES_HPP

