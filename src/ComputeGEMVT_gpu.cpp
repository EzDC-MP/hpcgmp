
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
 @file ComputeGEMVT_gpu.cpp

 HPCG routine for computing GEMV transpose (dot-products)
 */
#if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP) | defined(HPCG_WITH_KOKKOSKERNELS)

#ifndef HPCG_NO_MPI
 #include "Utils_MPI.hpp"
#endif

#include "DataTypes.hpp"
#include "ComputeGEMVT_ref.hpp"
#include "hpgmp.hpp"
#include "mytimer.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMVT_ref(const local_int_t m, const local_int_t n,
                     const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const Vector_type & x,
                     const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type & y) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename            Vector_type::scalar_type scalarX_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarY_type;

  assert(x.localLength >= m); // Test vector lengths
  assert(y.m >= n);
  assert(y.n == 1);

  // Output serial dense vector 
  scalarY_type * const yv = y.values;

#if defined(HPCG_DEBUG)
  const scalarA_type one  (1.0);
  const scalarA_type zero (0.0);

  // Input serial dense vector 
  scalarA_type * const Av = A.values;
  scalarX_type * const xv = x.values;

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

  scalarA_type * const d_Av = A.d_values;
  scalarX_type * const d_xv = x.d_values;
  scalarY_type * const d_yv = y.d_values;

  double t0; TICK();
  #if defined(HPCG_WITH_KOKKOSKERNELS) & !defined(KOKKOS_HALF_T_IS_FLOAT)
  {
    using execution_space = Kokkos::DefaultExecutionSpace;
    Kokkos::View<scalarA_type **, Kokkos::LayoutLeft, execution_space> A_view(d_Av, m, n);
    Kokkos::View<scalarX_type *,  Kokkos::LayoutLeft, execution_space> x_view(d_xv, m);
    Kokkos::View<scalarY_type *,  Kokkos::LayoutLeft, execution_space> y_view(d_yv, n);

    // Call GEMV
    KokkosBlas::gemv("T", alpha, A_view, x_view, beta, y_view);
    TIME(y.time1);

    // Copy output serial dense vector to host
    TICK();
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    Kokkos::View<scalarY_type *, Kokkos::LayoutLeft, host_execution_space> h_view(yv, n);
    Kokkos::deep_copy(h_view, y_view);
  }
  #elif defined(HPCG_WITH_CUDA)
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
  TIME(y.time1);

  // Copy input serial dense vector to host
  TICK();
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
  TIME(y.time1);

  // Copy output serial dense vector to host
  TICK();
  if (hipSuccess != hipMemcpy(yv, d_yv, n*sizeof(scalarX_type), hipMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_x\n" );
  }
  #endif

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  int size; // Number of MPI processes
  MPI_Comm_size(A.comm, &size);
  if (size > 1) {
      MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarY_type>::getType ();
      MPI_Op MPI_SCALAR_SUM = MpiTypeTraits<scalarY_type>::getSumOp ();
      MPI_Allreduce(MPI_IN_PLACE, yv, n, MPI_SCALAR_TYPE, MPI_SCALAR_SUM, A.comm);
  }
  TIME(y.time2);
#else
  y.time2 = 0.0;
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMVT_ref< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, Vector<double> const&, double, SerialDenseMatrix<double> &);

template
int ComputeGEMVT_ref< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, Vector<float> const&, float, SerialDenseMatrix<float> &);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !defined(KOKKOS_HALF_T_IS_FLOAT) // if arch does not support half, then half = float
template
int ComputeGEMVT_ref< MultiVector<half_t>, Vector<half_t>, SerialDenseMatrix<half_t> >
  (int, int, half_t, MultiVector<half_t> const&, Vector<half_t> const&, half_t, SerialDenseMatrix<half_t> &);

template
int ComputeGEMVT_ref< MultiVector<half_t>, Vector<half_t>, SerialDenseMatrix<float> >
  (int, int, half_t, MultiVector<half_t> const&, Vector<half_t> const&, float, SerialDenseMatrix<float> &);

template
int ComputeGEMVT_ref< MultiVector<half_t>, Vector<half_t>, SerialDenseMatrix<double> >
  (int, int, half_t, MultiVector<half_t> const&, Vector<half_t> const&, double, SerialDenseMatrix<double> &);
#endif
#endif
