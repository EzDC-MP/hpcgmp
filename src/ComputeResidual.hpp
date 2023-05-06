
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

#ifndef COMPUTERESIDUAL_HPP
#define COMPUTERESIDUAL_HPP
#include "Vector.hpp"

template<class Vector_type>
int ComputeResidual(const local_int_t n, const Vector_type & v1, const Vector_type & v2,
                    typename Vector_type::scalar_type & residual);

#endif // COMPUTERESIDUAL_HPP
