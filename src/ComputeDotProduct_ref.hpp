
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef COMPUTEDOTPRODUCT_REF_HPP
#define COMPUTEDOTPRODUCT_REF_HPP
#include "Vector.hpp"

template<class Vector_type, class scalar_type = typename Vector_type::scalar_type>
int ComputeDotProduct_ref(const local_int_t n,
                          const Vector_type & x,
                          const Vector_type & y,
                          scalar_type & result,
                          double & time_allreduce);

#endif // COMPUTEDOTPRODUCT_REF_HPP
