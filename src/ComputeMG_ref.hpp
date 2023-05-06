
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

#ifndef COMPUTEMG_REF_HPP
#define COMPUTEMG_REF_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<class SparseMatrix_type, class Vector_type>
int ComputeMG_ref(const SparseMatrix_type  & A, const Vector_type & r, Vector_type & x, bool symmetric = true);

#endif // COMPUTEMG_REF_HPP
