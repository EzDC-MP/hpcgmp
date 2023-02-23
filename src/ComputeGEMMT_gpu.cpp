
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
 @file ComputeGEMMT_gpu.cpp

 HPCG routine for computing GEMM transpose (dot-products)
 */
#if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)

#ifndef HPCG_NO_MPI
 #include "Utils_MPI.hpp"
#endif

#include "DataTypes.hpp"
#include "ComputeGEMMT_ref.hpp"
#include "hpgmp.hpp"
#include "mytimer.hpp"

template<class MultiVector_type, class SerialDenseMatrix_type>
int ComputeGEMMT_ref(const local_int_t m, const local_int_t n, const local_int_t k,
                     const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const MultiVector_type & B,
                     const typename SerialDenseMatrix_type::scalar_type beta, SerialDenseMatrix_type & C) {

  typedef typename       MultiVector_type::scalar_type scalarA_type;
  typedef typename SerialDenseMatrix_type::scalar_type scalarC_type;

  // Output serial dense vector 
  scalarC_type * const Cv = C.values;

  scalarA_type * const d_Av = A.d_values;
  scalarA_type * const d_Bv = B.d_values;
  scalarC_type * const d_Cv = C.d_values;

  double t0; TICK();
  #if defined(HPCG_WITH_KOKKOSKERNELS)
  {
    using execution_space = Kokkos::DefaultExecutionSpace;
    Kokkos::View<scalarA_type **, Kokkos::LayoutLeft, execution_space> A_view(d_Av, k, m);
    Kokkos::View<scalarA_type **, Kokkos::LayoutLeft, execution_space> B_view(d_Bv, k, n);
    Kokkos::View<scalarC_type **, Kokkos::LayoutLeft, execution_space> C_view(d_Cv, m, n);

    // Call GEMM
    KokkosBlas::gemm("T", "N", alpha, A_view, B_view, beta, C_view);
    TIME(C.time1);

    // Copy output serial dense vector to host
    TICK();
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    Kokkos::View<scalarC_type **, Kokkos::LayoutLeft, host_execution_space> H_view(Cv, m, n);
    Kokkos::deep_copy(H_view, C_view);
  }
  #elif defined(HPCG_WITH_CUDA)
  // Perform GEMM on device
  if (std::is_same<scalarX_type, double>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasDgemv(x.handle, CUBLAS_OP_T,
                                             m, n, k,
                                             (double*)&alpha, (double*)d_Av, k,
                                                              (double*)d_Bv, k,
                                             (double*)&beta,  (double*)d_Cv, m)){
      printf( " Failed cublasDgemv\n" );
    }
  } else if (std::is_same<scalarX_type, float>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasSgemv(x.handle, CUBLAS_OP_T,
                                             m, n, k,
                                             (float*)&alpha, (float*)d_Av, k,
                                                             (float*)d_Bv, k,
                                             (float*)&beta,  (float*)d_Cv, m)){
      printf( " Failed cublasSgemv\n" );
    }
  }
  TIME(y.time1);

  // Copy input serial dense vector to host
  TICK();
  if (cudaSuccess != cudaMemcpy(Cv, d_Cv, m*n*sizeof(scalarX_type), cudaMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_C\n" );
  }
  #elif defined(HPCG_WITH_HIP)
  // Perform GEMM on device
  if (std::is_same<scalarX_type, double>::value) {
    if (rocblas_status_success != rocblas_dgemv(x.handle,
                                                rocblas_operation_transpose,
                                                rocblas_operation_transpose,
                                                m, n, k,
                                                (double*)&alpha, (double*)d_Av, k,
                                                                 (double*)d_Bv, k,
                                                (double*)&beta,  (double*)d_Cv, m)){
      printf( " Failed rocblas_dgemv\n" );
    }
  } else if (std::is_same<scalarX_type, float>::value) {
    if (rocblas_status_success != rocblas_sgemv(x.handle,
                                                rocblas_operation_transpose,
                                                rocblas_operation_transpose,
                                                m, n, k,
                                                (float*)&alpha, (float*)d_Av, k,
                                                                (float*)d_Bv, k,
                                                (float*)&beta,  (float*)d_Cv, m)){
      printf( " Failed rocblas_sgemv\n" );
    }
  }
  TIME(y.time1);

  // Copy output serial dense vector to host
  TICK();
  if (hipSuccess != hipMemcpy(Cv, d_Cv, m*n*sizeof(scalarX_type), hipMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_C\n" );
  }
  #endif

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  int size; // Number of MPI processes
  MPI_Comm_size(A.comm, &size);
  if (size > 1) {
      MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarY_type>::getType ();
      MPI_Op MPI_SCALAR_SUM = MpiTypeTraits<scalarY_type>::getSumOp ();
      MPI_Allreduce(MPI_IN_PLACE, Cv, m*n, MPI_SCALAR_TYPE, MPI_SCALAR_SUM, A.comm);
  }
  TIME(C.time2);
#else
  C.time2 = 0.0;
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMMT_ref< MultiVector<double>, SerialDenseMatrix<double> >
  (int, int, int, double, MultiVector<double> const&, MultiVector<double> const&, double, SerialDenseMatrix<double> &);

template
int ComputeGEMMT_ref< MultiVector<float>, SerialDenseMatrix<float> >
  (int, int, int, float, MultiVector<float> const&, MultiVector<float> const&, float, SerialDenseMatrix<float> &);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !KOKKOS_HALF_T_IS_FLOAT // if arch does not support half, then half = float
template
int ComputeGEMMT_ref< MultiVector<half_t>, SerialDenseMatrix<half_t> >
  (int, int, int, half_t, MultiVector<half_t> const&, MultiVector<half_t> const&, half_t, SerialDenseMatrix<half_t> &);

template
int ComputeGEMMT_ref< MultiVector<half_t>, SerialDenseMatrix<float> >
  (int, int, int, half_t, MultiVector<half_t> const&, MultiVector<half_t> const&, float, SerialDenseMatrix<float> &);

template
int ComputeGEMMT_ref< MultiVector<half_t>, SerialDenseMatrix<double> >
  (int, int, int, half_t, MultiVector<half_t> const&, MultiVector<half_t> const&, double, SerialDenseMatrix<double> &);
#endif
#endif
