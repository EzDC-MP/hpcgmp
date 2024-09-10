
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Ichitaro Yamazaki         (iyamaza@sandia.gov)
// Sivasankaran Rajamanickam (srajama@sandia.gov)
// Piotr Luszczek            (luszczek@eecs.utk.edu)
// Jack Dongarra             (dongarra@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef COMPUTESPMV_KAHAN_HPP
#define COMPUTESPMV_KAHAN_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"

template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV_kahan(const SparseMatrix_type & A, Vector_type & x, Vector_type & y);

#endif  // COMPUTESPMV_KAHAN_HPP
