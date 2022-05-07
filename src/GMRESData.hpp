
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
  int restart_length;   //!< restart length
  SC tolerance;         //!< tolerance = reference residual norm 

  // from validation step
  int refNumIters;      //!< number of reference iterations
  int optNumIters;      //!< number of optimized iterations

  // from benchmark step
  int numOfCalls;       //!< number of calls
  double refTotalFlops; //
  double refTotalTime;  //
  double optTotalFlops; //
  double optTotalTime;  //
  double *flops;        //!< total, dot, axpy, ortho, spmv, reduce, precond
  double *times;        //!< total, dot, axpy, ortho, spmv, reduce, precond
};

#endif // CGDATA_HPP

