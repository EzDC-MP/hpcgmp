
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
 @file GMRES_IR.cpp

 GMRES-IR routine
 */

#include <fstream>
#include <cmath>

#include "hpgmp.hpp"

#include "GMRES_IR.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"
#include "ComputeTRSM.hpp"
#include "ComputeGEMV.hpp"
#include "ComputeGEMVT.hpp"


/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see GMRES_IR_ref()
*/
template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESData_type, class GMRESData_type2, class Vector_type, class TestGMRESData_type>
int GMRES_IR(const SparseMatrix_type & A, const SparseMatrix_type2 & A_lo,
             GMRESData_type & data, GMRESData_type2 & data_lo, const Vector_type & b_hi, Vector_type & x_hi,
             const int restart_length, const int max_iter, const typename SparseMatrix_type::scalar_type tolerance,
             int & niters, typename SparseMatrix_type::scalar_type & normr_hi, typename SparseMatrix_type::scalar_type & normr0_hi,
             bool doPreconditioning, bool verbose, TestGMRESData_type & test_data) {

  // (working) precision for outer loop
  typedef typename SparseMatrix_type::scalar_type scalar_type;
  typedef MultiVector<scalar_type> MultiVector_type;
  //typedef SerialDenseMatrix<scalar_type> SerialDenseMatrix_type;
  // (lower) precision for inner loop
  typedef typename SparseMatrix_type2::scalar_type scalar_type2;
  typedef MultiVector<scalar_type2> MultiVector_type2;
  //typedef SerialDenseMatrix<scalar_type2> SerialDenseMatrix_type;
  typedef Vector<scalar_type2> Vector_type2;
  // (lower) precision for storing projected matrix
  typedef typename GMRESData_type2::project_type project_type;
  typedef SerialDenseMatrix<project_type> SerialDenseMatrix_type;
  #ifdef HPCG_WITH_KOKKOSKERNELS
  using AT_hi = Kokkos::Details::ArithTraits<scalar_type>;
  using AT_lo = Kokkos::Details::ArithTraits<scalar_type2>;
  using AT_pr = Kokkos::Details::ArithTraits<project_type>;
  #endif

  double t_begin = mytimer();  // Start timing right away
  double start_t = 0.0, t0 = 0.0, t1 = 0.0, t1_ = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0, t6 = 0.0,
         t7 = 0.0, t8 = 0.0, t9 = 0.0;
  double t1_comp = 0.0, t1_comm = 0.0;

  // vectors/matrices in scalar_type2 (lower)
  const global_int_t ione  = 1;
  const global_int_t itwo  = 2;
  const global_int_t ifour = 4;
  const scalar_type2 one  (1.0);
  const scalar_type2 zero (0.0);
  const project_type one_pr  (1.0);
  const project_type zero_pr (0.0);
  project_type normr, normr0;
  project_type alpha = zero_pr, beta = zero_pr;

  local_int_t  nrow = A_lo.localNumberOfRows;
  global_int_t Nrow = A.totalNumberOfRows;
  //Vector_type2 & x = data_lo.w; // Intermediate solution vector
  Vector_type2 & r = data_lo.r; // Residual vector
  Vector_type2 & z = data_lo.z; // Preconditioned residual vector
  //Vector_type2 & p = data_lo.p; // Direction vector (in MPI mode ncol>=nrow)
  //Vector_type2 & Ap = data_lo.Ap;

  SerialDenseMatrix_type H;
  SerialDenseMatrix_type h;
  SerialDenseMatrix_type t;
  SerialDenseMatrix_type cs;
  SerialDenseMatrix_type ss;
  MultiVector_type2 Q;
  MultiVector_type2 P;
  Vector_type2 Qkm1;
  Vector_type2 Qk;
  Vector_type2 Qj;
  InitializeMatrix(H,  restart_length+1, restart_length);
  InitializeMatrix(h,  restart_length+1, 1);
  InitializeMatrix(t,  restart_length+1, 1);
  InitializeMatrix(cs, restart_length+1, 1);
  InitializeMatrix(ss, restart_length+1, 1);
  InitializeMultiVector(Q, nrow, restart_length+1, A.comm);

  // vectors in scalar_type (higher)
  const scalar_type zero_hi (0.0);
  const scalar_type one_hi  (1.0);
  Vector_type & r_hi = data.r; // Residual vector
  Vector_type & z_hi = data.z; // Preconditioned residual vector
  Vector_type & p_hi = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector_type & Ap_hi = data.Ap;

  if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

  int print_freq = 1;
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
  if (verbose && A.geom->rank==0) {
    HPCG_fout << std::endl << " Running GMRES_IR(" << restart_length
                           << ") with max-iters = " << max_iter
                           << ", tol = " << tolerance
                           << " and restart = " << restart_length
                           << (doPreconditioning ? " with precond" : " without precond")
                           << ", nrow = " << nrow << std::endl;
    if (std::is_same<scalar_type2, double>::value) 
      HPCG_fout << " Inner-Iter precision : double" << std::endl;
    if (std::is_same<scalar_type2, float>::value) 
      HPCG_fout << " Inner-Iter precision : float" << std::endl;
    #ifdef HPCG_WITH_KOKKOSKERNELS
    if (std::is_same<scalar_type2, half_t>::value) 
      HPCG_fout << " Inner-Iter precision : half_t" << std::endl;
    #endif
    if (std::is_same<project_type, double>::value) 
      HPCG_fout << " Projection precision : double" << std::endl;
    if (std::is_same<project_type, float>::value) 
      HPCG_fout << " Projection precision : float" << std::endl;
    #ifdef HPCG_WITH_KOKKOSKERNELS
    if (std::is_same<project_type, half_t>::value) 
      HPCG_fout << " Projection precision : half_t" << std::endl;
    #endif
  }
  double flops = 0.0;
  double flops_gmg  = 0.0;
  double flops_spmv = 0.0;
  double flops_orth = 0.0;
  global_int_t numSpMVs_MG = 1+(A.mgData->numberOfPresmootherSteps + A.mgData->numberOfPostsmootherSteps);
  niters = 0;
  bool converged = false;
  while (niters <= max_iter && !converged) {
    // > Compute residual vector (higher working precision)
    // p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x_hi, p_hi);
    TICK(); ComputeSPMV(A, p_hi, Ap_hi); flops_spmv += (2*A.totalNumberOfNonzeros); TOCK(t3); // Ap = A*p
    TICK(); ComputeWAXPBY(nrow, one_hi, b_hi, -one_hi, Ap_hi, r_hi, A.isWaxpbyOptimized); flops += (itwo*Nrow);  TOCK(t9); // r = b - Ax (x stored in p)
    TICK(); ComputeDotProduct(nrow, r_hi, r_hi, normr_hi, t4, A.isDotProductOptimized); flops += (itwo*Nrow); TOCK(t9);
    #ifdef HPCG_WITH_KOKKOSKERNELS
    normr_hi = AT_hi::sqrt(normr_hi);
    #else
    normr_hi = sqrt(normr_hi);
    #endif
    test_data.numOfSPCalls++;
    // Record initial residual for convergence testing
    if (niters == 0) {
      normr0 = normr_hi;
      normr0_hi = normr_hi;
    }
    normr = normr_hi;

    // Convergence check
    if (verbose && A.geom->rank==0) {
      HPCG_fout << "GMRES_IR Residual at the start of restart cycle = "
                << normr_hi << " / " << normr0_hi << " = " << normr_hi/normr0_hi
                << ", H(0,0) = " << normr << std::endl;
    }
    if (normr_hi/normr0_hi < tolerance) {
      converged = true;
      if (verbose && A.geom->rank==0) HPCG_fout << " > GMRES_IR converged " << std::endl;
      break;
    }

    // > Scale to the residual vector in working precision
    TICK(); ScaleVectorValue<Vector_type, scalar_type> (r_hi, one_hi/normr_hi); flops += Nrow; TOCK(t9);

    // > Copy r as the initial basis vector (lower precision)
    GetVector(Q, 0, Qj);
    CopyVector(r_hi, Qj);

    // do forward GS instead of symmetric GS
    bool symmetric = false;

    // Start restart cycle
    global_int_t k = 1;
    SetMatrixValue(t, 0, 0, normr);
    while (k <= restart_length && normr/normr0 >= tolerance) {
      GetVector(Q, k-1, Qkm1);
      GetVector(Q, k,   Qk);

      TICK();
      if (doPreconditioning) {
        z.time1 = z.time2 = 0.0;
        ComputeMG(A_lo, Qkm1, z, symmetric); flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros); // Apply preconditioner
        test_data.numOfMGCalls++;
        t7 += z.time1; t8 += z.time2;
      } else {
        CopyVector(Qkm1, z);       // copy r to z (no preconditioning)
      }
      TOCK(t5); // Preconditioner apply time

      // Qk = A*z
      TICK(); ComputeSPMV(A_lo, z, Qk); flops_spmv += (2*A.totalNumberOfNonzeros); TOCK(t3);
      test_data.numOfSPCalls++;

      // orthogonalize z against Q(:,0:k-1), using dots
      bool use_mgs = false;
      TICK();
      if (use_mgs) {
        // MGS2
        for (int j = 0; j < k; j++) {
          // get j-th column of Q
          GetVector(Q, j, Qj);

          alpha = zero_pr;
          for (int i = 0; i < 2; i++) {
            // beta = Qk'*Qj
            START_T(); ComputeDotProduct<Vector_type2, project_type>
                         (nrow, Qk, Qj, beta, t4, A.isDotProductOptimized); STOP_T(t1);

            // Qk = Qk - beta * Qj
            START_T(); ComputeWAXPBY(nrow, one, Qk, -beta, Qj, Qk, A.isWaxpbyOptimized); STOP_T(t2);
            alpha += beta;
          }
          SetMatrixValue(H, j, k-1, alpha);
        }
        flops_orth += (ifour*k*Nrow);
      } else {
        // CGS2
        GetMultiVector(Q, 0, k-1, P);
        START_T(); ComputeGEMVT (nrow, k,  one, P, Qk, zero_pr, h, A.isGemvOptimized); STOP_T(t1); // h = Q(1:k)'*q(k+1), mul and add in proj_type
        START_T(); ComputeGEMV  (nrow, k, -one, P, h,  one,    Qk, A.isGemvOptimized); STOP_T(t2); // h = Q(1:k)'*q(k+1), mul in proj, but add in low
        t1_comp += h.time1; t1_comm += h.time2;
        for(int i = 0; i < k; i++) {
          SetMatrixValue(H, i, k-1, h.values[i]);
        }
        flops_orth += (ifour*k*Nrow);
        // reorthogonalize
        START_T(); ComputeGEMVT (nrow, k,  one, P, Qk, zero_pr, h, A.isGemvOptimized); STOP_T(t1); // h = Q(1:k)'*q(k+1)
        START_T(); ComputeGEMV  (nrow, k, -one, P, h,  one,    Qk, A.isGemvOptimized); STOP_T(t2); // h = Q(1:k)'*q(k+1)
        t1_comp += h.time1; t1_comm += h.time2;
        for(int i = 0; i < k; i++) {
          AddMatrixValue(H, i, k-1, h.values[i]);
        }
        flops_orth += (ifour*k*Nrow);
      }
      // beta = norm(Qk)
      START_T(); ComputeDotProduct<Vector_type2, project_type>(nrow, Qk, Qk, beta, t4, A.isDotProductOptimized); STOP_T(t1_);
      flops_orth += (itwo*Nrow);
      #ifdef HPCG_WITH_KOKKOSKERNELS
      beta = AT_pr::sqrt(beta);
      #else
      beta = sqrt(beta);
      #endif

      // Qk = Qk / beta
      START_T(); ScaleVectorValue(Qk, one_pr/beta); STOP_T(t2);
      flops_orth += (Nrow);
      TOCK(t6); // Ortho time
      SetMatrixValue(H, k, k-1, beta);

      // Given's rotation
      for(int j = 0; j < k-1; j++){
        project_type cj = project_type(GetMatrixValue(cs, j, 0));
        project_type sj = project_type(GetMatrixValue(ss, j, 0));
        project_type h1 = project_type(GetMatrixValue(H, j,   k-1));
        project_type h2 = project_type(GetMatrixValue(H, j+1, k-1));

        SetMatrixValue(H, j+1, k-1, -sj * h1 + cj * h2);
        SetMatrixValue(H, j,   k-1,  cj * h1 + sj * h2);
      }

      project_type f = project_type(GetMatrixValue(H, k-1, k-1));
      project_type g = project_type(GetMatrixValue(H, k,   k-1));

      project_type f2 = f*f;
      project_type g2 = g*g;
      project_type fg2 = f2 + g2;
      #ifdef HPCG_WITH_KOKKOSKERNELS
      project_type D1 = one_pr / AT_pr::sqrt(f2*fg2);
      #else
      project_type D1 = one / sqrt(f2*fg2);
      #endif
      project_type cj = f2*D1;
      fg2 = fg2 * D1;
      project_type sj = f*D1*g;
      SetMatrixValue(H, k-1, k-1, f*fg2);
      SetMatrixValue(H, k,   k-1, zero_pr);

      project_type v1 = project_type(GetMatrixValue(t, k-1, 0));
      project_type v2 = -v1*sj;
      SetMatrixValue(t, k,   0, v2);
      SetMatrixValue(t, k-1, 0, v1*cj);

      SetMatrixValue(ss, k-1, 0, sj);
      SetMatrixValue(cs, k-1, 0, cj);

      #ifdef HPCG_WITH_KOKKOSKERNELS
      normr = AT_pr::abs(v2);
      #else
      normr = std::abs(v2);
      #endif
      if (verbose && (k%print_freq == 0 || k+1 == restart_length)) {
        #if 1
        {
          // compute current approximation
          CopyVector(x_hi, p_hi);                                 // using p_hi for x_hi
          for (int i=0; i < k; i++) h.values[i] = t.values[i];   // using h for t
          ComputeTRSM(k, one_pr, H, h);
          if (doPreconditioning) {
            //#define HPGMRES_IR_UPDATE_X_IN_HIGH
            #ifdef HPGMRES_IR_UPDATE_X_IN_HIGH
            ComputeGEMV(nrow, k, one, Q, h, zero_hi, r_hi, A.isGemvOptimized);          // r = Q*t (using h for t)
            ComputeMG(A_lo, r_hi, z_hi, symmetric);                                     // z = M*r
            ComputeWAXPBY(nrow, one_hi, p_hi, one_hi, z_hi, p_hi, A.isWaxpbyOptimized); // x += z
            #else
            ComputeGEMV(nrow, k, one, Q, h, zero, r, A.isGemvOptimized);             // r = Q*t (using h for t)
            ComputeMG(A_lo, r, z, symmetric);                                        // z = M*r
            ComputeWAXPBY(nrow, one_hi, p_hi, one, z, p_hi, A.isWaxpbyOptimized);    // x += z
            #endif
          } else {
            ComputeGEMV (nrow, k, one_hi, Q, h, one_hi, p_hi, A.isGemvOptimized);    // x += Q*t
          }
          // compute residual norm
          ComputeSPMV(A, p_hi, Ap_hi); // Ap = A*p
          ComputeWAXPBY(nrow, one_hi, b_hi, -one_hi, Ap_hi, r_hi, A.isWaxpbyOptimized); // r = b - Ax (x stored in p)
          ComputeDotProduct(nrow, r_hi, r_hi, normr_hi, t4, A.isDotProductOptimized);
          #ifdef HPCG_WITH_KOKKOSKERNELS
          normr_hi = AT_hi::sqrt(normr_hi);
          #else
          normr_hi = sqrt(normr_hi);
          #endif
        }
        project_type ortho_err (0.0);
        {
          GetMultiVector(Q, 0, k, P);
          for (int j=0; j<=k; j++) {
            GetVector(Q, j, Qk);
            ComputeGEMVT (nrow, k+1, one, P, Qk, zero_pr, h, A.isGemvOptimized);
            for (int i=0; i<=k; i++) {
              project_type error_i = (i == j ? h.values[i]-one_pr : h.values[i]);
              #ifdef HPCG_WITH_KOKKOSKERNELS
              error_i = AT_pr::abs(error_i);
              #else
              error_i = std::abs(error_i);
              #endif
              ortho_err = (error_i > ortho_err ? error_i : ortho_err);
	      //if (std::is_same<scalar_type, double>::value && std::is_same<project_type, float>::value && doPreconditioning) {
              //  if (verbose && A.geom->rank==0) HPCG_fout << "h(" << i << ", " << j << ")=" << h.values[i] << std::endl;
              //}
            }
          }
        }
        #endif
        if (verbose && A.geom->rank==0) {
          HPCG_fout << "GMRES_IR Iteration = "<< k << " (" << niters << ")   Scaled Computed Residual = "
                    << normr << " / " << normr0 << " = " << normr/normr0;
          #if 1
          HPCG_fout << " (True Residual = " << normr_hi << " / " << normr0_hi << " = " << normr_hi/normr0_hi << ")";
          HPCG_fout << "  Ortho Error = " << ortho_err;
          #endif
          HPCG_fout << std::endl;
        }
      }
      niters ++;
      k ++;
    } // end of restart-cycle

    // prepare to restart
    if (verbose && A.geom->rank==0)
      HPCG_fout << "GMRES_IR restart: k = "<< k << " (" << niters << ")" << std::endl;
    // > update x
    ComputeTRSM(k-1, one_pr, H, t);
    if (doPreconditioning) {
      #ifdef HPGMRES_IR_UPDATE_X_IN_HIGH
      ComputeGEMV (nrow, k-1, one, Q, t, zero_hi, r_hi, A.isGemvOptimized); flops += (itwo*Nrow*(k-ione)); // r = Q*t

      z.time1 = z.time2 = 0.0;
      TICK();
      ComputeMG(A, r_hi, z_hi, symmetric); flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros);    // z = M*r
      TOCK(t5); // Preconditioner apply time
      test_data.numOfMGCalls++;
      t7 += z.time1; t8 += z.time2;

      // mixed-precision
      TICK(); ComputeWAXPBY(nrow, one_hi, x_hi, one_hi, z_hi, x_hi, A.isWaxpbyOptimized); flops += (itwo*Nrow); TOCK(t9); // x += z
      #else
      ComputeGEMV (nrow, k-1, one, Q, t, zero, r, A.isGemvOptimized); flops += (itwo*Nrow*(k-ione)); // r = Q*t

      z.time1 = z.time2 = 0.0;
      TICK();
      ComputeMG(A_lo, r, z, symmetric); flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros);    // z = M*r
      TOCK(t5); // Preconditioner apply time
      test_data.numOfMGCalls++;
      t7 += z.time1; t8 += z.time2;

      // mixed-precision
      TICK(); ComputeWAXPBY(nrow, one_hi, x_hi, one, z, x_hi, A.isWaxpbyOptimized); flops += (itwo*Nrow); TOCK(t9); // x += z
      #endif
    } else {
      // mixed-precision
      ComputeGEMV (nrow, k-1, one_hi, Q, t, one_hi, x_hi, A.isGemvOptimized); flops += (itwo*Nrow*(k-ione)); // x += Q*t
    }
  } // end of outer-loop


  // Store times
  double tt = mytimer() - t_begin;
  if (test_data.times != NULL) {
    test_data.times[0] += tt;       // Total time. All done...
    test_data.times[1] += t1 + t1_; // dot-product time
    test_data.times[2] += t2;       // WAXPBY time
    test_data.times[3] += t6;       // Ortho
    test_data.times[4] += t3;       // SPMV time
    test_data.times[5] += t4;       // AllReduce time
    test_data.times[6] += t5;       // preconditioner apply time
    test_data.times[7] += t7;       // > SpTRSV for GS
    test_data.times[8] += t8;       // > SpMV for GS
    test_data.times[9] += t9;       // Vector update time

    test_data.times_comp[1] += t1_comp; // dot-product time
    test_data.times_comm[1] += t1_comm; // dot-product time
  }
  double flops_tot = flops + flops_gmg + flops_spmv + flops_orth;
  if (verbose && A.geom->rank==0) {
    HPCG_fout << " > nnz(A)  : " << A.totalNumberOfNonzeros << std::endl;
    HPCG_fout << " > nnz(MG) : " << A.totalNumberOfMGNonzeros << " (" << numSpMVs_MG << ")" << std::endl;
    HPCG_fout << " > SpMV : " << (flops_spmv / 1000000000.0) << " / " << t3 << " = "
                              << (flops_spmv / 1000000000.0) / t3 << " Gflop/s" << std::endl;
    HPCG_fout << " > GMG  : " << (flops_gmg  / 1000000000.0) << " / " << t5 << " = "
                              << (flops_gmg  / 1000000000.0) / t5 << " Gflop/s" << std::endl;
    HPCG_fout << " > Orth : " << (flops_orth / 1000000000.0) << " / " << t6 << " = "
                              << (flops_orth / 1000000000.0) / t6 << " Gflop/s" << std::endl;
    HPCG_fout << " > Total: " << (flops_tot  / 1000000000.0) << " / " << tt << " = "
                              << (flops_tot  / 1000000000.0) / tt << " Gflop/s" << std::endl;
    HPCG_fout << std::endl;
  }
  if (test_data.flops != NULL) {
    test_data.flops[0] += flops_tot;
    test_data.flops[1] += flops_gmg;
    test_data.flops[2] += flops_spmv;
    test_data.flops[3] += flops_orth;
  }
  DeleteDenseMatrix(H);
  DeleteDenseMatrix(t);
  DeleteDenseMatrix(h);
  DeleteDenseMatrix(cs);
  DeleteDenseMatrix(ss);
  DeleteMultiVector(Q);

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int GMRES_IR< SparseMatrix<double>, SparseMatrix<double>, GMRESData<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double> const&, SparseMatrix<double> const&, GMRESData<double>&, GMRESData<double>&,
   Vector<double> const&, Vector<double>&, const int, const int, double, int&, double&, double&, bool, bool,
   TestGMRESData<double>&);

template
int GMRES_IR< SparseMatrix<float>, SparseMatrix<float>, GMRESData<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
  (SparseMatrix<float> const&, SparseMatrix<float> const&, GMRESData<float>&, GMRESData<float>&,
   Vector<float> const&, Vector<float>&, const int, const int, float, int&, float&, float&, bool, bool,
   TestGMRESData<float>&);


// mixed
template
int GMRES_IR< SparseMatrix<double>, SparseMatrix<float>, GMRESData<double>, GMRESData<float>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double> const&, SparseMatrix<float> const&, GMRESData<double>&, GMRESData<float>&,
   Vector<double> const&, Vector<double>&, const int, const int, double, int&, double&, double&, bool, bool,
   TestGMRESData<double>&);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !KOKKOS_HALF_T_IS_FLOAT // if arch does not support half, then half = float
template
int GMRES_IR< SparseMatrix<double>, SparseMatrix<half_t>, GMRESData<double>, GMRESData<half_t, half_t>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double> const&, SparseMatrix<half_t> const&, GMRESData<double>&, GMRESData<half_t>&,
   Vector<double> const&, Vector<double>&, const int, const int, double, int&, double&, double&, bool, bool,
   TestGMRESData<double>&);

template
int GMRES_IR< SparseMatrix<double>, SparseMatrix<half_t>, GMRESData<double>, GMRESData<half_t, float>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double> const&, SparseMatrix<half_t> const&, GMRESData<double>&, GMRESData<half_t, float>&,
   Vector<double> const&, Vector<double>&, const int, const int, double, int&, double&, double&, bool, bool,
   TestGMRESData<double>&);

template
int GMRES_IR< SparseMatrix<double>, SparseMatrix<half_t>, GMRESData<double>, GMRESData<half_t, double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double> const&, SparseMatrix<half_t> const&, GMRESData<double>&, GMRESData<half_t, double>&,
   Vector<double> const&, Vector<double>&, const int, const int, double, int&, double&, double&, bool, bool,
   TestGMRESData<double>&);
#endif
