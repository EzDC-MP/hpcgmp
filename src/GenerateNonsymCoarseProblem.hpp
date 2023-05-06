
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

#ifndef GENERATE_NONSYM_COARSEPROBLEM_HPP
#define GENERATE_NONSYM_COARSEPROBLEM_HPP
#include "SparseMatrix.hpp"

template<class SparseMatrix_type>
void GenerateNonsymCoarseProblem(const SparseMatrix_type & A);

#endif // GENERATE_NONSYM_COARSEPROBLEM_HPP
