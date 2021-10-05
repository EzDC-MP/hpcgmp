
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

#ifndef COMPUTEWAXPBY_HPP
#define COMPUTEWAXPBY_HPP
#include "Vector.hpp"

template<class VectorX_type, class VectorY_type, class VectorW_type>
int ComputeWAXPBY(const local_int_t n,
                  const typename VectorX_type::scalar_type alpha,
                  const VectorX_type & x,
                  const typename VectorY_type::scalar_type beta,
                  const VectorY_type & y,
                        VectorW_type & w,
                  bool & isOptimized);

#endif // COMPUTEWAXPBY_HPP
