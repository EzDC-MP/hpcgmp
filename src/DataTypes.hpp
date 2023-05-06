
#ifndef HPGMP_DATA_TYPES_HPP
#define HPGMP_DATA_TYPES_HPP

#ifdef HPGMP_WITH_CUDA
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cublas_v2.h>
 #include <cusparse.h>
#elif defined(HPGMP_WITH_HIP)
 #include <hip/hip_runtime_api.h>
 #include <rocblas.h>
 #include <rocsparse.h>
#endif

#endif
