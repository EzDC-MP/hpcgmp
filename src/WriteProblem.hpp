
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

#ifndef WRITEPROBLEM_HPP
#define WRITEPROBLEM_HPP
#include "Geometry.hpp"
#include "SparseMatrix.hpp"

template<class SparseMatrix_type, class Vector_type>
int WriteProblem(const Geometry & geom, const SparseMatrix_type & A, const Vector_type b, const Vector_type x, const Vector_type xexact);

#endif // WRITEPROBLEM_HPP
