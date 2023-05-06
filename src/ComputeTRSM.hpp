
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
 @file Vector.hpp

 HPGMP data structures for dense vectors
 */

#ifndef COMPUTE_TRSM_HPP
#define COMPUTE_TRSM_HPP

#include "Geometry.hpp"
#include "SerialDenseMatrix.hpp"

template<class SerialDenseMatrix_type>
int ComputeTRSM(const local_int_t n,
                const typename SerialDenseMatrix_type::scalar_type alpha,
                const SerialDenseMatrix_type & U,
                      SerialDenseMatrix_type & x);

#endif // COMPUTE_TRSM_HPP
