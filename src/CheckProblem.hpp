
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

#ifndef CHECKPROBLEM_HPP
#define CHECKPROBLEM_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template <class SparseMatrix_type, class Vector_type>
void CheckProblem(SparseMatrix_type & A, Vector_type * b, Vector_type * x, Vector_type * xexact);

#endif // CHECKPROBLEM_HPP
