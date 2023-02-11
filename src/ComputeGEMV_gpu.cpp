
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
 @file ComputeGEMV_gpu.cpp

 HPCG routine for computing GEMV (vector-update)
 */
#if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)

#include "ComputeGEMV_ref.hpp"
#include "hpgmp.hpp"
#include "DataTypes.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMV_ref(const local_int_t m, const local_int_t n,
                    const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const SerialDenseMatrix_type & x,
                    const typename      Vector_type::scalar_type beta,  const Vector_type & y) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarX_type;
  typedef typename            Vector_type::scalar_type scalarY_type;

  const scalarY_type one  (1.0);
  const scalarY_type zero (0.0);

  assert(x.m >= n); // Test vector lengths
  assert(x.n == 1);
  assert(y.localLength >= m);

  // Input serial dense vector 
  scalarX_type * const xv = x.values;

  scalarA_type * const Av = A.values;
  scalarY_type * const yv = y.values;

#if defined(HPCG_DEBUG)
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

  scalarA_type * const d_Av = A.d_values;
  scalarX_type * const d_xv = x.d_values;
  scalarY_type * const d_yv = y.d_values;
  #if defined(HPCG_WITH_KOKKOSKERNELS)
  {
    using execution_space = Kokkos::DefaultExecutionSpace;
    Kokkos::View<scalarA_type **, Kokkos::LayoutLeft, execution_space> A_view(d_Av, m, n);
    Kokkos::View<scalarX_type *,  Kokkos::LayoutLeft, execution_space> x_view(d_xv, n);
    Kokkos::View<scalarY_type *,  Kokkos::LayoutLeft, execution_space> y_view(d_yv, m);
    // Copy input serial dense vector to device
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    Kokkos::View<scalarX_type *, Kokkos::LayoutLeft, host_execution_space> h_view(xv, n);
    Kokkos::deep_copy(x_view, h_view);

    // Call GEMV
    KokkosBlas::gemv("N", alpha, A_view, x_view, beta, y_view);
  }
  #else
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
    #endif
  } else {
    HPCG_vout << " Mixed-precision GEMV not supported" << std::endl;

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

#if defined(HPCG_WITH_KOKKOSKERNELS) & !defined(KOKKOS_HALF_T_IS_FLOAT) // if arch does not support half, then half = float
template
int ComputeGEMV_ref< MultiVector<half_t>, Vector<half_t>, SerialDenseMatrix<half_t> >
  (int, int, half_t, MultiVector<half_t> const&, SerialDenseMatrix<half_t> const&, half_t, Vector<half_t> const&);
#endif

// mixed
template
int ComputeGEMV_ref< MultiVector<float>, Vector<double>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, double, Vector<double> const&);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !defined(KOKKOS_HALF_T_IS_FLOAT) // if arch does not support half, then half = float
template
int ComputeGEMV_ref< MultiVector<half_t>, Vector<half_t>, SerialDenseMatrix<float> >
  (int, int, half_t, MultiVector<half_t> const&, SerialDenseMatrix<float> const&, half_t, Vector<half_t> const&);

template
int ComputeGEMV_ref< MultiVector<half_t>, Vector<double>, SerialDenseMatrix<float> >
  (int, int, half_t, MultiVector<half_t> const&, SerialDenseMatrix<float> const&, double, Vector<double> const&);

template
int ComputeGEMV_ref< MultiVector<half_t>, Vector<double>, SerialDenseMatrix<half_t> >
  (int, int, half_t, MultiVector<half_t> const&, SerialDenseMatrix<half_t> const&, double, Vector<double> const&);
#endif

#endif
