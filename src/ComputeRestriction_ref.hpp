
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

#ifndef COMPUTERESTRICTION_REF_HPP
#define COMPUTERESTRICTION_REF_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"

template<class SparseMatrix_type, class Vector_type>
int ComputeRestriction_ref(const SparseMatrix_type & A, const Vector_type & rf);

#endif // COMPUTERESTRICTION_REF_HPP
