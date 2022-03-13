
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

#ifdef HPCG_WITH_CUDA
 #include <cuda_runtime.h>
 #include <cublas_v2.h>
#elif defined(HPCG_WITH_HIP)
 #include <hip/hip_runtime_api.h>
 #include <rocblas.h>
#endif

#include "ComputeGEMV_ref.hpp"
#include "hpgmp.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMV_ref(const local_int_t m, const local_int_t n,
                    const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const SerialDenseMatrix_type & x,
                    const typename      Vector_type::scalar_type beta,  const Vector_type & y) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarX_type;
  typedef typename            Vector_type::scalar_type scalarY_type;

  const scalarA_type one  (1.0);
  const scalarA_type zero (0.0);

  assert(x.m >= n); // Test vector lengths
  assert(x.n == 1);
  assert(y.localLength >= m);

  // Input serial dense vector 
  const scalarX_type * const xv = x.values;

  scalarA_type * const Av = A.values;
  scalarY_type * const yv = y.values;

  #if 0//defined(HPCG_WITH_HIP)
  printf( " ** GEMV with HIP **\n" );
  if (hipSuccess != hipMemcpy(A.d_values, Av, m*n*sizeof(scalarA_type), hipMemcpyHostToDevice)) {
    printf( " Failed to memcpy d_y\n" );
  }
  if (hipSuccess != hipMemcpy(x.d_values, xv,   n*sizeof(scalarX_type), hipMemcpyHostToDevice)) {
    printf( " Failed to memcpy d_x\n" );
  }
  if (hipSuccess != hipMemcpy(y.d_values, yv,   m*sizeof(scalarY_type), hipMemcpyHostToDevice)) {
    printf( " Failed to memcpy d_y\n" );
  }
  #endif

#if (!defined(HPCG_WITH_CUDA) & !defined(HPCG_WITH_HIP)) | defined(HPCG_DEBUG)
  // GEMV on HOST CPU
  if (beta == zero) {
    for (local_int_t i = 0; i < m; i++) yv[i] = zero;
  } else if (beta != one) {
    for (local_int_t i = 0; i < m; i++) yv[i] *= beta;
  }

  if (alpha == one) {
    for (local_int_t j=0; j<n; j++)
      for (local_int_t i=0; i<m; i++) {
        yv[i] += Av[i + j*m] * xv[j];
    }
  } else {
    for (local_int_t j=0; j<n; j++)
      for (local_int_t i=0; i<m; i++) {
        yv[i] += alpha * Av[i + j*m] * xv[j];
    }
  }
#endif

#if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  scalarA_type * const d_Av = A.d_values;
  scalarX_type * const d_xv = x.d_values;
  scalarY_type * const d_yv = y.d_values;
  if ((std::is_same<scalarX_type, double>::value && std::is_same<scalarY_type, double>::value && std::is_same<scalarA_type, double>::value) ||
      (std::is_same<scalarX_type, float >::value && std::is_same<scalarY_type, float >::value && std::is_same<scalarA_type, float >::value)) {

    #if defined(HPCG_WITH_CUDA)
    // Copy input serial dense vector to device
    if (cudaSuccess != cudaMemcpy(d_xv, xv, n*sizeof(scalarX_type), cudaMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_x\n" );
    }

    // Perform GEMV on device
    if (std::is_same<scalarX_type, double>::value) {
      if (CUBLAS_STATUS_SUCCESS != cublasDgemv(y.handle, CUBLAS_OP_N,
                                               m, n,
                                               (double*)&alpha, (double*)d_Av, m,
                                                                (double*)d_xv, 1,
                                               (double*)&beta,  (double*)d_yv, 1)){
        printf( " Failed cublasDgemv\n" );
      }
    } else if (std::is_same<scalarX_type, float>::value) {
      if (CUBLAS_STATUS_SUCCESS != cublasSgemv(y.handle, CUBLAS_OP_N,
                                               m, n,
                                               (float*)&alpha, (float*)d_Av, m,
                                                               (float*)d_xv, 1,
                                               (float*)&beta,  (float*)d_yv, 1)){
        printf( " Failed cublasSgemv\n" );
      }
    }
    #elif defined(HPCG_WITH_HIP)
    // Copy input serial dense vector to device
    if (hipSuccess != hipMemcpy(d_xv, xv, n*sizeof(scalarX_type), hipMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_x\n" );
    }

    // Perform GEMV on device
    if (std::is_same<scalarX_type, double>::value) {
      if (rocblas_status_success != rocblas_dgemv(y.handle, rocblas_operation_none,
                                                  m, n,
                                                  (double*)&alpha, (double*)d_Av, m,
                                                                   (double*)d_xv, 1,
                                                  (double*)&beta,  (double*)d_yv, 1)){
        printf( " Failed rocblas_dgemv\n" );
      }
    } else if (std::is_same<scalarX_type, float>::value) {
      if (rocblas_status_success != rocblas_sgemv(y.handle, rocblas_operation_none,
                                                  m, n,
                                                  (float*)&alpha, (float*)d_Av, m,
                                                                  (float*)d_xv, 1,
                                                  (float*)&beta,  (float*)d_yv, 1)){
        printf( " Failed rocblas_sgemv\n" );
      }
    }
    #if 0 // TODO just for debug
    if (hipSuccess != hipMemcpy(yv, d_yv, m*sizeof(scalarY_type), hipMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_y\n" );
    }
    #endif
    #endif
  } else {
    HPCG_fout << " Mixed-precision GEMV not supported" << std::endl;

    // Copy input matrix A from HOST CPU
    #if defined(HPCG_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(Av, d_Av, m*n*sizeof(scalarA_type), cudaMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_y\n" );
    }
    if (beta != zero) {
      if (cudaSuccess != cudaMemcpy(yv, d_yv, m*sizeof(scalarY_type), cudaMemcpyDeviceToHost)) {
        printf( " Failed to memcpy d_y\n" );
      }
    }
    #elif defined(HPCG_WITH_HIP)
    if (hipSuccess != hipMemcpy(Av, d_Av, m*n*sizeof(scalarA_type), hipMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_y\n" );
    }
    if (beta != zero) {
      if (hipSuccess != hipMemcpy(yv, d_yv, m*sizeof(scalarY_type), hipMemcpyDeviceToHost)) {
        printf( " Failed to memcpy d_y\n" );
      }
    }
    #endif

    // GEMV on HOST CPU
    if (beta == zero) {
      for (local_int_t i = 0; i < m; i++) yv[i] = zero;
    } else if (beta != one) {
      for (local_int_t i = 0; i < m; i++) yv[i] *= beta;
    }

    if (alpha == one) {
      for (local_int_t i=0; i<m; i++) {
        for (local_int_t j=0; j<n; j++)
          yv[i] += Av[i + j*m] * xv[j];
      }
    } else {
      for (local_int_t i=0; i<m; i++) {
        for (local_int_t j=0; j<n; j++)
          yv[i] += alpha * Av[i + j*m] * xv[j];
      }
    }

    // Copy output vector Y from HOST CPU
    #if defined(HPCG_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(d_yv, yv, m*sizeof(scalarY_type), cudaMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_y\n" );
    }
    #elif defined(HPCG_WITH_HIP)
    if (hipSuccess != hipMemcpy(d_yv, yv, m*sizeof(scalarY_type), hipMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_y\n" );
    }
    #endif
  }
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMV_ref< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, SerialDenseMatrix<double> const&, double, Vector<double> const&);

template
int ComputeGEMV_ref< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, float, Vector<float> const&);


// mixed
template
int ComputeGEMV_ref< MultiVector<float>, Vector<double>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, double, Vector<double> const&);

