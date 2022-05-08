
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#ifdef HPCG_WITH_CUDA
 #include <cuda_runtime.h>
 #include <cublas_v2.h>
#elif defined(HPCG_WITH_HIP)
 #include <hip/hip_runtime_api.h>
 #include <rocblas.h>
#endif

#include "OptimizeProblem.hpp"
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
template<class SparseMatrix_type, class GMRESData_type, class Vector_type>
int OptimizeProblem(SparseMatrix_type & A, GMRESData_type & data, Vector_type & b, Vector_type & x, Vector_type & xexact) {

  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints

#if defined(HPCG_USE_MULTICOLORING)
  const local_int_t nrow = A.localNumberOfRows;
  std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  int totalColors = 1;
  colors[0] = 0; // first point gets color 0

  // Finds colors in a greedy (a likely non-optimal) fashion.

  for (local_int_t i=1; i < nrow; ++i) {
    if (colors[i] == nrow) { // if color not assigned
      std::vector<int> assigned(totalColors, 0);
      int currentlyAssigned = 0;
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
        local_int_t curCol = currentColIndices[j];
        if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
          if (assigned[colors[curCol]] == 0)
            currentlyAssigned += 1;
          assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
        } // else // could take advantage of indices being sorted
      }

      if (currentlyAssigned < totalColors) { // if there is at least one color left to use
        for (int j=0; j < totalColors; ++j)  // try all current colors
          if (assigned[j] == 0) { // if no neighbor with this color
            colors[i] = j;
            break;
          }
      } else {
        if (colors[i] == nrow) {
          colors[i] = totalColors;
          totalColors += 1;
        }
      }
    }
  }

  std::vector<local_int_t> counters(totalColors);
  for (local_int_t i=0; i<nrow; ++i)
    counters[colors[i]]++;

  // form in-place prefix scan
  local_int_t old=counters[0], old0;
  for (local_int_t i=1; i < totalColors; ++i) {
    old0 = counters[i];
    counters[i] = counters[i-1] + old;
    old = old0;
  }
  counters[0] = 0;

  // translate `colors' into a permutation
  for (local_int_t i=0; i<nrow; ++i) // for each color `c'
    colors[i] = counters[colors[i]]++;
#endif

#if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  {
    typedef typename SparseMatrix_type::scalar_type SC;

    SparseMatrix_type * curLevelMatrix = &A;
    do {
      // -------------------------
      // form CSR on host
      const local_int_t nrow = curLevelMatrix->localNumberOfRows;
      const local_int_t ncol = curLevelMatrix->localNumberOfColumns;
      global_int_t nnzL = 0;
      global_int_t nnz = curLevelMatrix->localNumberOfNonzeros;
      int *h_row_ptr = (int*)malloc((nrow+1)* sizeof(int));
      int *h_col_ind = (int*)malloc( nnz    * sizeof(int));
      SC  *h_nzvals  = (SC *)malloc( nnz    * sizeof(SC));

      nnz = 0;
      h_row_ptr[0] = 0;
      for (local_int_t i=0; i<nrow; i++)  {
        const SC * const cur_vals = curLevelMatrix->matrixValues[i];
        const local_int_t * const cur_inds = curLevelMatrix->mtxIndL[i];

        const int cur_nnz = curLevelMatrix->nonzerosInRow[i];
        for (int j=0; j<cur_nnz; j++) {
          h_nzvals[nnz+j] = cur_vals[j];
          h_col_ind[nnz+j] = cur_inds[j];
          if (cur_inds[j] <= i) {
            nnzL++;
          }
        }
        nnz += cur_nnz;
        h_row_ptr[i+1] = nnz;
      }

      // copy CSR(A) to device
      #if defined(HPCG_WITH_CUDA)
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_row_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_row_ptr(nrow=%d)\n",nrow );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_col_idx), nnz*sizeof(int))) {
        printf( " Failed to allocate A.d_col_idx(nnz=%lld)\n",nnz );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_nzvals),  nnz*sizeof(SC))) {
        printf( " Failed to allocate A.d_nzvals(nnz=%lld)\n",nnz );
      }

      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_row_ptr, h_row_ptr, (nrow+1)*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_row_ptr\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_col_idx, h_col_ind, nnz*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_col_idx\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_nzvals,  h_nzvals,  nnz*sizeof(SC),  cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_nzvals\n" );
      }
      #elif defined(HPCG_WITH_HIP)
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_row_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_row_ptr(nrow=%d)\n",nrow );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_col_idx), nnz*sizeof(int))) {
        printf( " Failed to allocate A.d_col_idx(nnz=%lld)\n",nnz );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_nzvals),  nnz*sizeof(SC))) {
        printf( " Failed to allocate A.d_nzvals(nnz=%lld)\n",nnz );
      }

      if (hipSuccess != hipMemcpy(curLevelMatrix->d_row_ptr, h_row_ptr, (nrow+1)*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_row_ptr\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_col_idx, h_col_ind, nnz*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_col_idx\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_nzvals,  h_nzvals,  nnz*sizeof(SC), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_nzvals\n" );
      }
      #endif

      // free matrix on host
      free(h_row_ptr);
      free(h_col_ind);
      free(h_nzvals);

      // -------------------------
      // Extract lower/upper-triangular matrix
      global_int_t nnzU = nnz-nnzL;
      int *h_Lrow_ptr = (int*)malloc((nrow+1)* sizeof(int));
      int *h_Lcol_ind = (int*)malloc( nnzL   * sizeof(int));
      SC  *h_Lnzvals  = (SC *)malloc( nnzL   * sizeof(SC));
      int *h_Urow_ptr = (int*)malloc((nrow+1)* sizeof(int));
      int *h_Ucol_ind = (int*)malloc( nnzU   * sizeof(int));
      SC  *h_Unzvals  = (SC *)malloc( nnzU   * sizeof(SC));
      nnzL = 0;
      nnzU = 0;
      h_Lrow_ptr[0] = 0;
      h_Urow_ptr[0] = 0;
      for (local_int_t i=0; i<nrow; i++)  {
        const SC * const cur_vals = curLevelMatrix->matrixValues[i];
        const local_int_t * const cur_inds = curLevelMatrix->mtxIndL[i];

        const int cur_nnz = curLevelMatrix->nonzerosInRow[i];
        for (int j=0; j<cur_nnz; j++) {
          if (cur_inds[j] <= i) {
            h_Lnzvals[nnzL] = cur_vals[j];
            h_Lcol_ind[nnzL] = cur_inds[j];
            nnzL ++;
          } else {
            h_Unzvals[nnzU] = cur_vals[j];
            h_Ucol_ind[nnzU] = cur_inds[j];
            nnzU ++;
          }
        }
        h_Lrow_ptr[i+1] = nnzL;
        h_Urow_ptr[i+1] = nnzU;
      }
      curLevelMatrix->nnzL = nnzL;
      curLevelMatrix->nnzU = nnzU;

      // copy CSR(L) to device
      #if defined(HPCG_WITH_CUDA)
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Lrow_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_Lrow_ptr\n" );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Lcol_idx), nnzL*sizeof(int))) {
        printf( " Failed to allocate A.d_Lcol_idx\n" );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Lnzvals),  nnzL*sizeof(SC))) {
        printf( " Failed to allocate A.d_Lrow_ptr\n" );
      }

      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Lrow_ptr, h_Lrow_ptr, (nrow+1)*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lrow_ptr\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Lcol_idx, h_Lcol_ind, nnzL*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lcol_idx\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Lnzvals,  h_Lnzvals,  nnzL*sizeof(SC),  cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lrow_ptr\n" );
      }
      #elif defined(HPCG_WITH_HIP)
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Lrow_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_Lrow_ptr\n" );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Lcol_idx), nnzL*sizeof(int))) {
        printf( " Failed to allocate A.d_Lcol_idx\n" );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Lnzvals),  nnzL*sizeof(SC))) {
        printf( " Failed to allocate A.d_Lrow_ptr\n" );
      }

      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Lrow_ptr, h_Lrow_ptr, (nrow+1)*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lrow_ptr\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Lcol_idx, h_Lcol_ind, nnzL*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lcol_idx\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Lnzvals,  h_Lnzvals,  nnzL*sizeof(SC),  hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Lrow_ptr\n" );
      }
      #endif

      // copy CSR(U) to device
      #if defined(HPCG_WITH_CUDA)
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Urow_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_Urow_ptr(nrow=%d)\n",nrow );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Ucol_idx), nnzU*sizeof(int))) {
        printf( " Failed to allocate A.d_Ucol_idx(nnzU=%d)\n",nnzU );
      }
      if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->d_Unzvals),  nnzU*sizeof(SC))) {
        printf( " Failed to allocate A.d_Urow_ptr(nnzU=%d)\n",nnzU );
      }

      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Urow_ptr, h_Urow_ptr, (nrow+1)*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Urow_ptr\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Ucol_idx, h_Ucol_ind, nnzU*sizeof(int), cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Ucol_idx\n" );
      }
      if (cudaSuccess != cudaMemcpy(curLevelMatrix->d_Unzvals,  h_Unzvals,  nnzU*sizeof(SC),  cudaMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Urow_ptr\n" );
      }
      #elif defined(HPCG_WITH_HIP)
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Urow_ptr), (nrow+1)*sizeof(int))) {
        printf( " Failed to allocate A.d_Urow_ptr(nrow=%d)\n",nrow );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Ucol_idx), nnzU*sizeof(int))) {
        printf( " Failed to allocate A.d_Ucol_idx(nnzU=%lld)\n",nnzU );
      }
      if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->d_Unzvals),  nnzU*sizeof(SC))) {
        printf( " Failed to allocate A.d_Urow_ptr(nnzU=%lld)\n",nnzU );
      }

      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Urow_ptr, h_Urow_ptr, (nrow+1)*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Urow_ptr\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Ucol_idx, h_Ucol_ind, nnzU*sizeof(int), hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Ucol_idx\n" );
      }
      if (hipSuccess != hipMemcpy(curLevelMatrix->d_Unzvals,  h_Unzvals,  nnzU*sizeof(SC),  hipMemcpyHostToDevice)) {
        printf( " Failed to memcpy A.d_Urow_ptr\n" );
      }
      #endif

      // free matrix on host
      free(h_Lrow_ptr);
      free(h_Lcol_ind);
      free(h_Lnzvals);
      free(h_Urow_ptr);
      free(h_Ucol_ind);
      free(h_Unzvals);

      // for debuging, TODO: remove these
      InitializeVector(curLevelMatrix->x, nrow, curLevelMatrix->comm);
      InitializeVector(curLevelMatrix->y, ncol, curLevelMatrix->comm);
 
      #if defined(HPCG_WITH_CUDA)
      // -------------------------
      // create Handle (for each matrix)
      cusparseCreate(&(curLevelMatrix->cusparseHandle));

      // -------------------------
      // descriptor for A
      cusparseCreateMatDescr(&(curLevelMatrix->descrA));
      cusparseSetMatType(curLevelMatrix->descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(curLevelMatrix->descrA, CUSPARSE_INDEX_BASE_ZERO);

      // -------------------------
      // run analysis for triangular solve
      cusparseCreateMatDescr(&(curLevelMatrix->descrL));
      cusparseCreateSolveAnalysisInfo(&(curLevelMatrix->infoL));
      cusparseSetMatType(curLevelMatrix->descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
      cusparseSetMatIndexBase(curLevelMatrix->descrL, CUSPARSE_INDEX_BASE_ZERO);
      if (std::is_same<SC, double>::value) {
        cusparseDcsrsv_analysis(curLevelMatrix->cusparseHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, nrow, nnzL,
                                curLevelMatrix->descrL,
                                (double *)curLevelMatrix->d_Lnzvals, curLevelMatrix->d_Lrow_ptr, curLevelMatrix->d_Lcol_idx,
                                curLevelMatrix->infoL);
      } else if (std::is_same<SC, float>::value) {
        cusparseScsrsv_analysis(curLevelMatrix->cusparseHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, nrow, nnzL,
                                curLevelMatrix->descrL,
                                (float *)curLevelMatrix->d_Lnzvals, curLevelMatrix->d_Lrow_ptr, curLevelMatrix->d_Lcol_idx,
                                curLevelMatrix->infoL);
      }

      // -------------------------
      // descriptor for U
      cusparseCreateMatDescr(&(curLevelMatrix->descrU));
      cusparseSetMatType(curLevelMatrix->descrU, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(curLevelMatrix->descrU, CUSPARSE_INDEX_BASE_ZERO);
      #elif defined(HPCG_WITH_HIP)
      // -------------------------
      // create Handle (for each matrix)
      rocsparse_create_handle(&(curLevelMatrix->rocsparseHandle));

      // -------------------------
      // descriptor for A
      rocsparse_datatype rocsparse_compute_type = rocsparse_datatype_f64_r;
      if (std::is_same<SC, float>::value) {
        rocsparse_compute_type = rocsparse_datatype_f32_r;
      }
      rocsparse_create_csr_descr(&(curLevelMatrix->descrA), nrow, ncol, nnz,
                                 curLevelMatrix->d_row_ptr, curLevelMatrix->d_col_idx, curLevelMatrix->d_nzvals,
                                 rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_compute_type);
      const SC one  (1.0);
      const SC zero (0.0);
      curLevelMatrix->buffer_size_A = 0;
      curLevelMatrix->buffer_A = nullptr;
      rocsparse_dnvec_descr vecX, vecY; 
      rocsparse_create_dnvec_descr(&vecX, ncol, (void*)curLevelMatrix->x.d_values, rocsparse_compute_type);
      rocsparse_create_dnvec_descr(&vecY, nrow, (void*)curLevelMatrix->y.d_values, rocsparse_compute_type);
      rocsparse_spmv(curLevelMatrix->rocsparseHandle, rocsparse_operation_none,
                     &one, curLevelMatrix->descrA, vecX, &zero, vecY, 
                     rocsparse_compute_type, rocsparse_spmv_alg_default,
                     &curLevelMatrix->buffer_size_A, curLevelMatrix->buffer_A);
      if (curLevelMatrix->buffer_size_A <= 0) curLevelMatrix->buffer_size_A = 1;
      hipMalloc(&curLevelMatrix->buffer_A, curLevelMatrix->buffer_size_A);

      // -------------------------
      // run analysis for triangular solve
      rocsparse_create_csr_descr(&(curLevelMatrix->descrL), nrow, nrow, nnzL,
                                 curLevelMatrix->d_Lrow_ptr, curLevelMatrix->d_Lcol_idx, curLevelMatrix->d_Lnzvals,
                                 rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_compute_type);
      curLevelMatrix->buffer_size_L = 0;
      curLevelMatrix->buffer_L = nullptr;
      rocsparse_create_dnvec_descr(&vecX, nrow, (void*)curLevelMatrix->x.d_values, rocsparse_compute_type);
      rocsparse_create_dnvec_descr(&vecY, nrow, (void*)curLevelMatrix->y.d_values, rocsparse_compute_type);
      rocsparse_status spsv_stat = rocsparse_spsv(curLevelMatrix->rocsparseHandle, rocsparse_operation_none,
                                                  &one, curLevelMatrix->descrL, vecY, vecY, rocsparse_compute_type, 
                                                  rocsparse_spsv_alg_default, rocsparse_spsv_stage_buffer_size,
                                                  &curLevelMatrix->buffer_size_L, curLevelMatrix->buffer_L);
      if (rocsparse_status_success != spsv_stat) {
        printf( " Failed rocsparse_spsv(buffer size, stat = %d)\n",spsv_stat );
      }
      //if (curLevelMatrix->buffer_size_L <= 0) curLevelMatrix->buffer_size_L = 1;
      hipMalloc(&curLevelMatrix->buffer_L, curLevelMatrix->buffer_size_L);
      spsv_stat = rocsparse_spsv(curLevelMatrix->rocsparseHandle, rocsparse_operation_none,
                                 &one, curLevelMatrix->descrL, vecY, vecY, rocsparse_compute_type, 
                                 rocsparse_spsv_alg_default, rocsparse_spsv_stage_preprocess,
                                 &curLevelMatrix->buffer_size_L, curLevelMatrix->buffer_L);
      if (rocsparse_status_success != spsv_stat) {
        printf( " Failed rocsparse_spsv(preprocess, stat = %d)\n",spsv_stat );
      }

      // -------------------------
      // descriptor for U
      rocsparse_create_csr_descr(&(curLevelMatrix->descrU), nrow, ncol, nnzU,
                                 curLevelMatrix->d_Urow_ptr, curLevelMatrix->d_Ucol_idx, curLevelMatrix->d_Unzvals,
                                 rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_compute_type);
      const SC alpha (-1.0);
      const SC beta   (1.0);
      curLevelMatrix->buffer_size_U = 0;
      curLevelMatrix->buffer_U = nullptr;
      rocsparse_create_dnvec_descr(&vecX, ncol, (void*)curLevelMatrix->x.d_values, rocsparse_compute_type);
      rocsparse_create_dnvec_descr(&vecY, nrow, (void*)curLevelMatrix->y.d_values, rocsparse_compute_type);
      rocsparse_spmv(curLevelMatrix->rocsparseHandle, rocsparse_operation_none,
                     &alpha, curLevelMatrix->descrU, vecX, &beta, vecY, 
                     rocsparse_compute_type, rocsparse_spmv_alg_default,
                     &curLevelMatrix->buffer_size_U, curLevelMatrix->buffer_U);
      if (curLevelMatrix->buffer_size_U <= 0) curLevelMatrix->buffer_size_U = 1;
      hipMalloc(&curLevelMatrix->buffer_U, curLevelMatrix->buffer_size_U);
      #endif

      if (curLevelMatrix->mgData!=0) {
        // -------------------------
        // store restriction as CRS
        local_int_t * f2c = curLevelMatrix->mgData->f2cOperator;
        local_int_t nc = curLevelMatrix->mgData->rc->localLength;
        h_row_ptr = (int*)malloc((nc+1) * sizeof(int));
        h_col_ind = (int*)malloc( nc    * sizeof(int));
        h_nzvals  = (SC *)malloc( nc    * sizeof(SC));

        h_row_ptr[0] = 0;
        for (local_int_t i=0; i<nc; ++i) {
          h_col_ind[i] = f2c[i];
          h_nzvals[i] = 1.0;
          h_row_ptr[i+1] = i+1;
        }

        // copy CSR(R) to device
        #if defined(HPCG_WITH_CUDA)
        if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->mgData->d_row_ptr), (nc+1)*sizeof(int))) {
          printf( " Failed to allocate A.d_row_ptr(nc=%d)\n",nc );
        }
        if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->mgData->d_col_idx), nc*sizeof(int))) {
          printf( " Failed to allocate A.d_col_idx(nc=%d)\n",nc );
        }
        if (cudaSuccess != cudaMalloc ((void**)&(curLevelMatrix->mgData->d_nzvals),  nc*sizeof(SC))) {
          printf( " Failed to allocate A.d_nzvals(nc=%d)\n",nc );
        }

        if (cudaSuccess != cudaMemcpy(curLevelMatrix->mgData->d_row_ptr, h_row_ptr, (nc+1)*sizeof(int), cudaMemcpyHostToDevice)) {
          printf( " Failed to memcpy A.d_row_ptr\n" );
        }
        if (cudaSuccess != cudaMemcpy(curLevelMatrix->mgData->d_col_idx, h_col_ind, nc*sizeof(int), cudaMemcpyHostToDevice)) {
          printf( " Failed to memcpy A.d_col_idx\n" );
        }
        if (cudaSuccess != cudaMemcpy(curLevelMatrix->mgData->d_nzvals,  h_nzvals,  nc*sizeof(SC),  cudaMemcpyHostToDevice)) {
          printf( " Failed to memcpy A.d_nzvals\n" );
        }
        #elif defined(HPCG_WITH_HIP)
        if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->mgData->d_row_ptr), (nc+1)*sizeof(int))) {
          printf( " Failed to allocate A.d_row_ptr(nc=%d)\n",nc );
        }
        if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->mgData->d_col_idx), nc*sizeof(int))) {
          printf( " Failed to allocate A.d_col_idx(nc=%d)\n",nc );
        }
        if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->mgData->d_nzvals),  nc*sizeof(SC))) {
          printf( " Failed to allocate A.d_nzvals(nc=%d)\n",nc );
        }

        if (hipSuccess != hipMemcpy(curLevelMatrix->mgData->d_row_ptr, h_row_ptr, (nc+1)*sizeof(int), hipMemcpyHostToDevice)) {
          printf( " Failed to memcpy A.d_row_ptr\n" );
        }
        if (hipSuccess != hipMemcpy(curLevelMatrix->mgData->d_col_idx, h_col_ind, nc*sizeof(int), hipMemcpyHostToDevice)) {
          printf( " Failed to memcpy A.d_col_idx\n" );
        }
        if (hipSuccess != hipMemcpy(curLevelMatrix->mgData->d_nzvals,  h_nzvals,  nc*sizeof(SC),  hipMemcpyHostToDevice)) {
          printf( " Failed to memcpy A.d_nzvals\n" );
        }

	// store explicity store transpose
        free(h_row_ptr);
        h_row_ptr = (int*)calloc((nrow+1), sizeof(int));
        for (local_int_t i=0; i<nc; ++i) {
          h_row_ptr[f2c[i]+1] = 1;
        }
        for (local_int_t i=0; i<nrow; ++i) {
          h_row_ptr[i+1] += h_row_ptr[i];
        }

        for (local_int_t i=0; i<nc; ++i) {
          h_col_ind[h_row_ptr[f2c[i]]] = i;
          h_nzvals[h_row_ptr[f2c[i]]] = 1.0;
        }

        if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->mgData->d_tran_row_ptr), (nrow+1)*sizeof(int))) {
          printf( " Failed to allocate A.d_row_ptr(nc=%d)\n",nc );
        }
        if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->mgData->d_tran_col_idx), nc*sizeof(int))) {
          printf( " Failed to allocate A.d_col_idx(nc=%d)\n",nc );
        }
        if (hipSuccess != hipMalloc ((void**)&(curLevelMatrix->mgData->d_tran_nzvals),  nc*sizeof(SC))) {
          printf( " Failed to allocate A.d_nzvals(nc=%d)\n",nc );
        }

        if (hipSuccess != hipMemcpy(curLevelMatrix->mgData->d_tran_row_ptr, h_row_ptr, (nrow+1)*sizeof(int), hipMemcpyHostToDevice)) {
          printf( " Failed to memcpy A.d_row_ptr\n" );
        }
        if (hipSuccess != hipMemcpy(curLevelMatrix->mgData->d_tran_col_idx, h_col_ind, nc*sizeof(int), hipMemcpyHostToDevice)) {
          printf( " Failed to memcpy A.d_col_idx\n" );
        }
        if (hipSuccess != hipMemcpy(curLevelMatrix->mgData->d_tran_nzvals,  h_nzvals,  nc*sizeof(SC),  hipMemcpyHostToDevice)) {
          printf( " Failed to memcpy A.d_nzvals\n" );
        }
        #endif

        // -------------------------
        // descriptor for restrictor
        #if defined(HPCG_WITH_CUDA)
        cusparseCreateMatDescr(&(curLevelMatrix->mgData->descrR));
        cusparseSetMatType(curLevelMatrix->mgData->descrR, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(curLevelMatrix->mgData->descrR, CUSPARSE_INDEX_BASE_ZERO);
        #elif defined(HPCG_WITH_HIP)
        rocsparse_create_csr_descr(&(curLevelMatrix->mgData->descrR), nc, nrow, nc,
                                   curLevelMatrix->mgData->d_row_ptr, curLevelMatrix->mgData->d_col_idx, curLevelMatrix->mgData->d_nzvals,
                                   rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_compute_type);
        curLevelMatrix->mgData->buffer_size_R = 0;
        curLevelMatrix->mgData->buffer_R = nullptr;
        rocsparse_create_dnvec_descr(&vecX, nrow, (void*)curLevelMatrix->x.d_values, rocsparse_compute_type);
        rocsparse_create_dnvec_descr(&vecY, nc,   (void*)curLevelMatrix->y.d_values, rocsparse_compute_type);
        rocsparse_spmv(curLevelMatrix->rocsparseHandle, rocsparse_operation_none,
                       &one, curLevelMatrix->mgData->descrR, vecX, &one, vecY, 
                       rocsparse_compute_type, rocsparse_spmv_alg_default,
                       &curLevelMatrix->mgData->buffer_size_R, curLevelMatrix->mgData->buffer_R);
        if (curLevelMatrix->mgData->buffer_size_R <= 0) curLevelMatrix->mgData->buffer_size_R = 1;
        hipMalloc(&curLevelMatrix->mgData->buffer_R, curLevelMatrix->mgData->buffer_size_R);
        //
        rocsparse_create_csr_descr(&(curLevelMatrix->mgData->descrP), nrow, nc, nc,
                                   curLevelMatrix->mgData->d_tran_row_ptr, curLevelMatrix->mgData->d_tran_col_idx, curLevelMatrix->mgData->d_tran_nzvals,
                                   rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_compute_type);
        curLevelMatrix->mgData->buffer_size_P = 0;
        curLevelMatrix->mgData->buffer_P = nullptr;
        rocsparse_spmv(curLevelMatrix->rocsparseHandle, rocsparse_operation_none,
                       &one, curLevelMatrix->mgData->descrP, vecY, &one, vecX, 
                       rocsparse_compute_type, rocsparse_spmv_alg_default,
                       &curLevelMatrix->mgData->buffer_size_P, curLevelMatrix->mgData->buffer_P);
        if (curLevelMatrix->mgData->buffer_size_P <= 0) curLevelMatrix->mgData->buffer_size_P = 1;
        hipMalloc(&curLevelMatrix->mgData->buffer_P, curLevelMatrix->mgData->buffer_size_P);
        #endif

        // free matrix on host
        free(h_row_ptr);
        free(h_col_ind);
        free(h_nzvals);
      } //A.mgData!=0

      // next matrix
      curLevelMatrix = curLevelMatrix->Ac;
    } while (curLevelMatrix != 0);
  }
  {
    typedef typename Vector_type::scalar_type vector_SC;
    #if defined(HPCG_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(b.d_values,  b.values, (b.localLength)*sizeof(vector_SC),  cudaMemcpyHostToDevice)) {
      printf( " Failed to memcpy b\n" );
    }
    if (cudaSuccess != cudaMemcpy(x.d_values,  x.values, (x.localLength)*sizeof(vector_SC),  cudaMemcpyHostToDevice)) {
      printf( " Failed to memcpy x\n" );
    }
    #elif defined(HPCG_WITH_HIP)
    if (hipSuccess != hipMemcpy(b.d_values,  b.values, (b.localLength)*sizeof(vector_SC),  hipMemcpyHostToDevice)) {
      printf( " Failed to memcpy b\n" );
    }
    if (hipSuccess != hipMemcpy(x.d_values,  x.values, (x.localLength)*sizeof(vector_SC),  hipMemcpyHostToDevice)) {
      printf( " Failed to memcpy x\n" );
    }
    #endif
  }
#endif

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
template<class SparseMatrix_type>
double OptimizeProblemMemoryUse(const SparseMatrix_type & A) {

  return 0.0;

}


/* --------------- *
 * specializations *
 * --------------- */

template
int OptimizeProblem< SparseMatrix<double>, GMRESData<double>, Vector<double> >
  (SparseMatrix<double>&, GMRESData<double>&, Vector<double>&, Vector<double>&, Vector<double>&);

template
double OptimizeProblemMemoryUse< SparseMatrix<double> >
  (SparseMatrix<double> const&);

template
int OptimizeProblem< SparseMatrix<float>, GMRESData<float>, Vector<float> >
  (SparseMatrix<float>&, GMRESData<float>&, Vector<float>&, Vector<float>&, Vector<float>&);

template
double OptimizeProblemMemoryUse< SparseMatrix<float> >
  (SparseMatrix<float> const&);

// mixed-precision
template
int OptimizeProblem< SparseMatrix<float>, GMRESData<double>, Vector<double> >
  (SparseMatrix<float>&, GMRESData<double>&, Vector<double>&, Vector<double>&, Vector<double>&);

