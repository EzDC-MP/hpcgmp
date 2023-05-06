
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file TestGMRES.hpp

 HPGMRES data structure
 */

#ifndef TESTGMRES_HPP
#define TESTGMRES_HPP

#include "hpgmp.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"
#include "GMRESData.hpp"

template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESData_type, class GMRESData_type2, class Vector_type, class TestGMRESData_type>
extern int TestGMRES(SparseMatrix_type & A, SparseMatrix_type2 & A_lo, GMRESData_type & data, GMRESData_type2 & data_lo,
                     Vector_type & b, Vector_type & x, bool test_diagonal_exaggeration, bool test_noprecond, TestGMRESData_type & test_data);

template<class SparseMatrix_type, class GMRESData_type, class Vector_type, class TestGMRESData_type>
extern int TestGMRES(SparseMatrix_type & A, GMRESData_type & data, Vector_type & b, Vector_type & x,
                     bool test_diagonal_exaggeration, bool test_noprecond, TestGMRESData_type & test_data);

#endif  // TESTGMRES_HPP

