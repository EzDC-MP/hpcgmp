
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Ichitaro Yamazaki         (iyamaza@sandia.gov)
// Sivasankaran Rajamanickam (srajama@sandia.gov)
// Piotr Luszczek            (luszczek@eecs.utk.edu)
// Jack Dongarra             (dongarra@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file MGData.hpp

 HPGMP data structure
 */

#ifndef MGDATA_HPP
#define MGDATA_HPP

#include <cassert>
#include "DataTypes.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<class SC>
class MGData {
public:
  typedef Vector<SC> Vector_type;
  int numberOfPresmootherSteps; // Call ComputeSYMGS this many times prior to coarsening
  int numberOfPostsmootherSteps; // Call ComputeSYMGS this many times after coarsening
  local_int_t * f2cOperator; //!< 1D array containing the fine operator local IDs that will be injected into coarse space.
  Vector_type * rc; // coarse grid residual vector
  Vector_type * xc; // coarse grid solution vector
  Vector_type * Axf; // fine grid residual vector
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;
  size_t buffer_size_R;
  size_t buffer_size_P;
  void* buffer_R;
  void* buffer_P;
  #if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)
  // to store the restrictiion as CRS matrix on device
  int *d_row_ptr;
  int *d_col_idx;
  SC  *d_nzvals;   //!< values of matrix entries
  #if defined(HPGMP_WITH_CUDA)
  cusparseMatDescr_t descrR;
  #elif defined(HPGMP_WITH_HIP)
  rocsparse_spmat_descr descrR;

  // to store transpose
  rocsparse_spmat_descr descrP;
  int *d_tran_row_ptr;
  int *d_tran_col_idx;
  SC  *d_tran_nzvals;   //!< values of matrix entries
  #endif
  #endif
};

/*!
 Constructor for the data structure of CG vectors.

 @param[in] Ac - Fully-formed coarse matrix
 @param[in] f2cOperator -
 @param[out] data the data structure for CG vectors that will be allocated to get it ready for use in CG iterations
 */
template <class MGData_type>
inline void InitializeMGData(local_int_t * f2cOperator,
                             typename MGData_type::Vector_type * rc,
                             typename MGData_type::Vector_type * xc,
                             typename MGData_type::Vector_type * Axf,
                             MGData_type & data) {
  data.numberOfPresmootherSteps = 1;
  data.numberOfPostsmootherSteps = 1;
  data.f2cOperator = f2cOperator; // Space for injection operator
  data.rc = rc;
  data.xc = xc;
  data.Axf = Axf;
  return;
}

/*!
 Destructor for the CG vectors data.

 @param[inout] data the MG data structure whose storage is deallocated
 */
template <class MGData_type>
inline void DeleteMGData(MGData_type & data) {

  delete [] data.f2cOperator;
  DeleteVector(*data.Axf);
  DeleteVector(*data.rc);
  DeleteVector(*data.xc);
  delete data.Axf;
  delete data.rc;
  delete data.xc;
  return;
}

#endif // MGDATA_HPP

