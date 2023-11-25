
#ifndef HPCG_DATA_TYPES_HPP
#define HPCG_DATA_TYPES_HPP

#ifdef HPCG_WITH_KOKKOSKERNELS
#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_gauss_seidel.hpp>

#if defined(KOKKOS_HALF_T_IS_FLOAT)
typedef float half_t;
#else
typedef Kokkos::Experimental::half_t half_t;
#endif

#ifndef HPCG_NO_MPI
#include "mpi.h"
extern MPI_Datatype    HPCG_MPI_HALF;
extern MPI_Op          MPI_SUM_HALF;
extern void HPCG_FP16_SUM_FUNCTION(void* invec, void* inoutvec, int *len, MPI_Datatype *datatype);
#endif
#endif

#ifdef HPCG_WITH_CUDA
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cublas_v2.h>
 #include <cusparse.h>
#elif defined(HPCG_WITH_HIP)
 #include <hip/hip_runtime_api.h>
 #include <rocblas.h>
 #include <rocsparse.h>

#define GpuErrorCheck(call, msg)                                                                  \
do{                                                                                               \
    hipError_t Err = call;                                                                        \
    if(hipSuccess != Err){                                                                        \
        printf("Error - %s : %s at %d: '%s'\n", msg, __FILE__, __LINE__, hipGetErrorString(Err)); \
        exit(0);                                                                                  \
    }                                                                                             \
}while(0)


#endif

//#ifdef HPCG_WITH_KOKKOSKERNELS
//#include <Kokkos_Half.hpp>
//#endif
#endif
