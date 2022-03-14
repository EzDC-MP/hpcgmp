
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
 @file Vector.hpp

 HPCG data structures for dense vectors
 */
#ifndef HPCG_NO_MPI
 #include "Utils_MPI.hpp"
#endif
#ifdef HPCG_WITH_CUDA
 #include <cuda_runtime.h>
 #include <cublas_v2.h>
#elif defined(HPCG_WITH_HIP)
 #include <hip/hip_runtime_api.h>
 #include <rocblas.h>
#endif

#include "ComputeGEMVT_ref.hpp"
#include "hpgmp.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMVT_ref(const local_int_t m, const local_int_t n,
                     const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const Vector_type & x,
                     const typename      Vector_type::scalar_type beta,  const SerialDenseMatrix_type & y) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarX_type;
  typedef typename            Vector_type::scalar_type scalarY_type;

  const scalarA_type one  (1.0);
  const scalarA_type zero (0.0);

  assert(x.localLength >= m); // Test vector lengths
  assert(y.m >= n);
  assert(y.n == 1);

  // Input serial dense vector 
  scalarA_type * const Av = A.values;
  scalarX_type * const xv = x.values;
  scalarY_type * const yv = y.values;

#if (!defined(HPCG_WITH_CUDA) & !defined(HPCG_WITH_HIP)) | defined(HPCG_DEBUG)
  // GEMV on HOST CPU
  if (beta == zero) {
    for (local_int_t i = 0; i < n; i++) yv[i] = zero;
  } else if (beta != one) {
    for (local_int_t i = 0; i < n; i++) yv[i] *= beta;
  }

  if (alpha == one) {
    for (local_int_t j=0; j<n; j++)
      for (local_int_t i=0; i<m; i++) {
        yv[j] += Av[i + j*m] * xv[i];
    }
  } else {
    for (local_int_t i=0; i<m; i++) {
      for (local_int_t j=0; j<n; j++)
        yv[j] += alpha * Av[i + j*m] * xv[i];
    }
  }
#endif

#if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  scalarA_type * const d_Av = A.d_values;
  scalarX_type * const d_xv = x.d_values;
  scalarY_type * const d_yv = y.d_values;

  #if defined(HPCG_WITH_CUDA)
  // Perform GEMV on device
  if (std::is_same<scalarX_type, double>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasDgemv(x.handle, CUBLAS_OP_T,
                                             m, n,
                                             (double*)&alpha, (double*)d_Av, m,
                                                              (double*)d_xv, 1,
                                             (double*)&beta,  (double*)d_yv, 1)){
      printf( " Failed cublasDgemv\n" );
    }
  } else if (std::is_same<scalarX_type, float>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasSgemv(x.handle, CUBLAS_OP_T,
                                             m, n,
                                             (float*)&alpha, (float*)d_Av, m,
                                                             (float*)d_xv, 1,
                                             (float*)&beta,  (float*)d_yv, 1)){
      printf( " Failed cublasSgemv\n" );
    }
  }

  // Copy input serial dense vector to host
  if (cudaSuccess != cudaMemcpy(yv, d_yv, n*sizeof(scalarX_type), cudaMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_x\n" );
  }
  #elif defined(HPCG_WITH_HIP)
  // Perform GEMV on device
  if (std::is_same<scalarX_type, double>::value) {
    if (rocblas_status_success != rocblas_dgemv(x.handle, rocblas_operation_transpose,
                                                m, n,
                                                (double*)&alpha, (double*)d_Av, m,
                                                                 (double*)d_xv, 1,
                                                (double*)&beta,  (double*)d_yv, 1)){
      printf( " Failed rocblas_dgemv\n" );
    }
  } else if (std::is_same<scalarX_type, float>::value) {
    if (rocblas_status_success != rocblas_sgemv(x.handle, rocblas_operation_transpose,
                                                m, n,
                                                (float*)&alpha, (float*)d_Av, m,
                                                                (float*)d_xv, 1,
                                                (float*)&beta,  (float*)d_yv, 1)){
      printf( " Failed rocblas_sgemv\n" );
    }
  }

  // Copy input serial dense vector to host
  if (hipSuccess != hipMemcpy(yv, d_yv, n*sizeof(scalarX_type), hipMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_x\n" );
  }
  #endif
#endif

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarY_type>::getType ();
  MPI_Allreduce(MPI_IN_PLACE, yv, n, MPI_SCALAR_TYPE, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMVT_ref< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, Vector<double> const&, double, SerialDenseMatrix<double> const&);

template
int ComputeGEMVT_ref< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, Vector<float> const&, float, SerialDenseMatrix<float> const&);


