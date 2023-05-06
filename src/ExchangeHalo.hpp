
//@HEADER
// ***************************************************
//
// HPGMP:High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef EXCHANGEHALO_HPP
#define EXCHANGEHALO_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<class SparseMatrix_type, class Vector_type>
void ExchangeHalo(const SparseMatrix_type & A, Vector_type & x);

#endif // EXCHANGEHALO_HPP
