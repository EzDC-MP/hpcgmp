
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

#ifndef GENERATE_NONSYM_PROBLEM_V1_REF_HPP
#define GENERATE_NONSYM_PROBLEM_V1_REF_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<class SparseMatrix_type, class Vector_type>
void GenerateNonsymProblem_v1_ref(SparseMatrix_type & A, Vector_type * b, Vector_type * x, Vector_type * xexact, bool init_vect = true);

#endif // GENERATEPROBLEM_REF_HPP
