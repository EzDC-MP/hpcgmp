
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

#ifndef GENERATE_NONSYM_PROBLEM_HPP
#define GENERATE_NONSYM_PROBLEM_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<class SparseMatrix_type, class Vector_type>
void GenerateNonsymProblem(SparseMatrix_type & A, Vector_type * b, Vector_type * x, Vector_type * xexact, bool init_vect = true);

#endif // GENERATEPROBLEM_HPP
