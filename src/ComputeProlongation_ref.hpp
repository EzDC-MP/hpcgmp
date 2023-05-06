
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

#ifndef COMPUTEPROLONGATION_REF_HPP
#define COMPUTEPROLONGATION_REF_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"

template<class SparseMatrix_type, class Vector_type>
int ComputeProlongation_ref(const SparseMatrix_type & Af, Vector_type & xf);

#endif // COMPUTEPROLONGATION_REF_HPP
