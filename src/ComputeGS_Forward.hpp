
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

#ifndef COMPUTEGS_FORWARD_HPP
#define COMPUTEGS_FORWARD_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<class SparseMatrix_type, class Vector_type>
int ComputeGS_Forward(const SparseMatrix_type & A, const Vector_type & r, Vector_type & x);

#endif // COMPUTEGS_FORWARD_HPP
