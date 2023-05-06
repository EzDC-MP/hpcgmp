
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

#ifndef SETUPHALO_REF_HPP
#define SETUPHALO_REF_HPP
#include "SparseMatrix.hpp"

template<class SparseMatrix_type>
void SetupHalo_ref(SparseMatrix_type & A);

#endif // SETUPHALO_REF_HPP
