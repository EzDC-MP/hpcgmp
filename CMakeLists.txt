#
# HPCG Benchmark CMake configuration
#
cmake_minimum_required( VERSION 3.0 FATAL_ERROR )

project( hpgmp
    VERSION 3.1
    LANGUAGES CXX
    DESCRIPTION "High Performance Conjugate Gradient Benchmark (HPCG) " )

#
# Options
#
option(HPCG_ENABLE_CONTIGUOUS_ARRAYS "Enable contiguous arrays for better cache pre-fetch" OFF)
option(HPCG_ENABLE_CUBIC_RADICAL_SEARCH "Enable faster search for optimal 3D process grid" OFF)
option(HPCG_ENABLE_DEBUG "Enable debug build" OFF)
option(HPCG_ENABLE_DETAILED_DEBUG "Enable detailed debug build" OFF)
option(HPCG_ENABLE_MPI "Enable MPI support" OFF)
option(HPCG_ENABLE_LONG_LONG "Enable use of 'long long' type for global indices" ON)
option(HPCG_ENABLE_OPENMP "Enable OpenMP support" OFF)

add_executable( xhpgmp src/main_hpgmp.cpp
    src/GMRES.cpp src/GMRES_IR.cpp src/TestGMRES.cpp
    src/ComputeResidual.cpp src/ExchangeHalo.cpp src/GenerateGeometry.cpp
    src/GenerateNonsymProblem.cpp src/GenerateNonsymProblem_v1_ref.cpp src/CheckProblem.cpp
    src/OptimizeProblem.cpp src/ReadHpcgDat.cpp src/ReportResults.cpp
    src/SetupHalo.cpp src/SetupHalo_ref.cpp
    src/SetupMatrix.cpp src/SetupProblem.cpp
    src/ValidGMRES.cpp src/BenchGMRES.cpp
    src/WriteProblem.cpp
    src/YAML_Doc.cpp src/YAML_Element.cpp 
    src/ComputeDotProduct.cpp src/ComputeDotProduct_ref.cpp src/ComputeDotProduct_gpu.cpp src/ComputeDotProduct_blas.cpp
    src/ComputeTRSM.cpp
    src/ComputeGEMV.cpp src/ComputeGEMV_ref.cpp src/ComputeGEMV_blas.cpp src/ComputeGEMV_gpu.cpp
    src/ComputeGEMVT.cpp src/ComputeGEMVT_ref.cpp src/ComputeGEMVT_blas.cpp src/ComputeGEMVT_gpu.cpp
    src/finalize.cpp src/init.cpp src/mytimer.cpp
    src/ComputeSPMV.cpp src/ComputeSPMV_ref.cpp src/ComputeSPMV_gpu.cpp src/ComputeSPMV_ssl2.cpp src/ComputeSPMV_kk.cpp
    src/ComputeSYMGS.cpp src/ComputeSYMGS_ref.cpp
    src/ComputeGS_Forward.cpp src/ComputeGS_Forward_ref.cpp src/ComputeGS_Forward_gpu.cpp src/ComputeGS_Forward_kk.cpp
    src/ComputeWAXPBY.cpp src/ComputeWAXPBY_ref.cpp src/ComputeWAXPBY_gpu.cpp
    src/ComputeMG.cpp src/ComputeMG_ref.cpp
    src/ComputeProlongation_ref.cpp src/ComputeRestriction_ref.cpp
    src/ComputeProlongation_gpu.cpp src/ComputeRestriction_gpu.cpp
    src/GenerateNonsymCoarseProblem.cpp
    src/ComputeOptimalShapeXYZ.cpp src/MixedBaseCounter.cpp
    src/CheckAspectRatio.cpp src/OutputFile.cpp)

add_executable( xhpgmp_time src/main_time.cpp
    src/GMRES.cpp src/GMRES_IR.cpp src/TestGMRES.cpp
    src/ComputeResidual.cpp src/ExchangeHalo.cpp src/GenerateGeometry.cpp
    src/GenerateNonsymProblem.cpp src/GenerateNonsymProblem_v1_ref.cpp src/CheckProblem.cpp
    src/OptimizeProblem.cpp src/ReadHpcgDat.cpp src/ReportResults.cpp
    src/SetupHalo.cpp src/SetupHalo_ref.cpp
    src/SetupMatrix.cpp src/SetupProblem.cpp
    src/ValidGMRES.cpp src/BenchGMRES.cpp
    src/WriteProblem.cpp
    src/YAML_Doc.cpp src/YAML_Element.cpp 
    src/ComputeDotProduct.cpp src/ComputeDotProduct_ref.cpp src/ComputeDotProduct_gpu.cpp src/ComputeDotProduct_blas.cpp
    src/ComputeTRSM.cpp
    src/ComputeGEMV.cpp src/ComputeGEMV_ref.cpp src/ComputeGEMV_blas.cpp src/ComputeGEMV_gpu.cpp
    src/ComputeGEMVT.cpp src/ComputeGEMVT_ref.cpp src/ComputeGEMVT_blas.cpp src/ComputeGEMVT_gpu.cpp
    src/finalize.cpp src/init.cpp src/mytimer.cpp
    src/ComputeSPMV.cpp src/ComputeSPMV_ref.cpp src/ComputeSPMV_gpu.cpp src/ComputeSPMV_ssl2.cpp src/ComputeSPMV_kk.cpp
    src/ComputeSYMGS.cpp src/ComputeSYMGS_ref.cpp
    src/ComputeGS_Forward.cpp src/ComputeGS_Forward_ref.cpp src/ComputeGS_Forward_gpu.cpp src/ComputeGS_Forward_kk.cpp
    src/ComputeWAXPBY.cpp src/ComputeWAXPBY_ref.cpp src/ComputeWAXPBY_gpu.cpp
    src/ComputeMG.cpp src/ComputeMG_ref.cpp
    src/ComputeProlongation_ref.cpp src/ComputeRestriction_ref.cpp
    src/ComputeProlongation_gpu.cpp src/ComputeRestriction_gpu.cpp
    src/GenerateNonsymCoarseProblem.cpp
    src/ComputeOptimalShapeXYZ.cpp src/MixedBaseCounter.cpp
    src/CheckAspectRatio.cpp src/OutputFile.cpp)

if (HPCG_ENABLE_CONTIGUOUS_ARRAYS)
    target_compile_definitions(xhpgmp PRIVATE HPCG_CONTIGUOUS_ARRAYS)
endif ()

if (HPCG_ENABLE_CUBIC_RADICAL_SEARCH)
    target_compile_definitions(xhpgmp PRIVATE HPCG_CUBIC_RADICAL_SEARCH)
endif ()

if (HPCG_ENABLE_DEBUG)
    target_compile_definitions(xhpgmp PRIVATE HPCG_DEBUG)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_DEBUG)
endif ()

if (HPCG_ENABLE_DETAILED_DEBUG)
    target_compile_definitions(xhpgmp PRIVATE HPCG_DETAILED_DEBUG)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_DETAILED_DEBUG)
endif ()

if (HPCG_ENABLE_MPI)
    set(MPI_CXX_SKIP_MPICXX ON)
    find_package(MPI REQUIRED)
    target_link_libraries(xhpgmp ${MPI_CXX_LIBRARIES})
    target_link_libraries(xhpgmp_time ${MPI_CXX_LIBRARIES})
else ()
    target_compile_definitions(xhpgmp PRIVATE HPCG_NO_MPI)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_NO_MPI)
endif ()

if (NOT HPCG_ENABLE_LONG_LONG)
    target_compile_definitions(xhpgmp PRIVATE HPCG_NO_LONG_LONG)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_NO_LONG_LONG)
endif ()

if (HPCG_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(xhpgmp ${OpenMP_CXX_LIBRARIES})
    target_link_libraries(xhpgmp_time ${OpenMP_CXX_LIBRARIES})
else ()
    target_compile_definitions(xhpgmp PRIVATE HPCG_NO_OPENMP)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_NO_OPENMP)
endif ()

if (HPCG_ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    target_compile_definitions(xhpgmp PRIVATE HPCG_WITH_CUDA)
    target_compile_definitions(xhpgmp PRIVATE HPCG_USE_FENCE)
    target_link_libraries(xhpgmp ${CUDA_LIBRARIES})

    target_compile_definitions(xhpgmp_time PRIVATE HPCG_WITH_CUDA)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_USE_FENCE)
    target_link_libraries(xhpgmp_time ${CUDA_LIBRARIES})
else ()
    target_compile_definitions(xhpgmp PRIVATE HPCG_NO_CUDA)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_NO_CUDA)
endif ()


if (HPCG_ENABLE_BLAS)
    target_compile_definitions(xhpgmp PRIVATE HPCG_WITH_BLAS)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_WITH_BLAS)
endif ()

if (HPCG_ENABLE_KOKKOSKERNELS)
    find_package(KokkosKernels REQUIRED)
    target_compile_definitions(xhpgmp PRIVATE HPCG_WITH_KOKKOSKERNELS)
    target_link_libraries(xhpgmp Kokkos::kokkoskernels)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_WITH_KOKKOSKERNELS)
    target_link_libraries(xhpgmp_time Kokkos::kokkoskernels)
else ()
    target_compile_definitions(xhpgmp PRIVATE HPCG_NO_KOKKOSKERNELS)
    target_compile_definitions(xhpgmp_time PRIVATE HPCG_NO_KOKKOSKERNELS)
endif ()
