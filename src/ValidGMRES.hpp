
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
 @file ValidGMRES.hpp

 HPGMRES data structure
 */

#ifndef VALIDGMRES_HPP
#define VALIDGMRES_HPP

#include "hpgmp.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"

template<class scalar_type, class scalar_type2, class TestGMRESData_type>
extern int ValidGMRES(int argc, char **argv, comm_type comm, int numberOfMgLevels, bool verbose, TestGMRESData_type & testcg_data);

#endif  // BENCHGMRES_HPP

