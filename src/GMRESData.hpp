
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
 @file CGData.hpp

 HPCG data structure
 */

#ifndef GMRESDATA_HPP
#define GMRESDATA_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

template <class SC>
class GMRESData {
public:
  Vector<SC> r; //!< pointer to residual vector
  Vector<SC> z; //!< pointer to preconditioned residual vector
  Vector<SC> p; //!< pointer to direction vector
  Vector<SC> w; //!< pointer to workspace
  Vector<SC> Ap; //!< pointer to Krylov vector
};

/*!
 Constructor for the data structure of GMRES vectors.

 @param[in]  A    the data structure that describes the problem matrix and its structure
 @param[out] data the data structure for GMRES  vectors that will be allocated to get it ready for use in GMRES iterations
 */
template <class SparseMatrix_type, class GMRESData_type>
inline void InitializeSparseGMRESData(SparseMatrix_type & A, GMRESData_type & data) {
  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;
  InitializeVector(data.r, nrow);
  InitializeVector(data.z, ncol);
  InitializeVector(data.p, ncol);
  InitializeVector(data.w, nrow);
  InitializeVector(data.Ap, nrow);
  return;
}

/*!
 Destructor for the GMRES vectors data.

 @param[inout] data the GMRES vectors data structure whose storage is deallocated
 */
template <class GMRESData_type>
inline void DeleteGMRESData(GMRESData_type & data) {

  DeleteVector (data.r);
  DeleteVector (data.z);
  DeleteVector (data.p);
  DeleteVector (data.w);
  DeleteVector (data.Ap);
  return;
}



template<class SC>
class TestGMRESData {
public:
  int ref_iters;   //!< number of reference iterations
  SC tolerance;    //!< tolerance = reference residual norm 

  int numOfCalls;     //!< number of calls
  int restart_length; //!< restart length
  int count_pass;     //!< number of succesful tests
  int count_fail;     //!< number of succesful tests
  int niters_max;     //!< max number of iterations
  int *niters;        //!< number of iterations
  double *times;      //!< total, dot, axpy, ortho, spmv, reduce, precond
  double *flops;      //!< total, gmg, spmv, ortho
  SC *normr0;         //!< initial residual norm
  SC *normr;          //!< residual norm achieved during test CG iterations
};

#endif // CGDATA_HPP

