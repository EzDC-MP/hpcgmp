
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

/*!
 @file ComputeGEMMT.hpp

 HPCG data structures for dense vectors
 */

#ifndef COMPUTE_GEMMT_HPP
#define COMPUTE_GEMMT_HPP

#include "Geometry.hpp"
#include "MultiVector.hpp"
#include "Vector.hpp"
#include "SerialDenseMatrix.hpp"

template<class MultiVector_type, class SerialDenseMatrix_type>
int ComputeGEMMT(const local_int_t m, const local_int_t n, const local_int_t k,
                const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const MultiVector_type & B,
                const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type & C,
                bool & isOptimized);

#endif // COMPUTE_GEMMT
