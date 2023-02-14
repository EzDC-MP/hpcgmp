
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
template<class VectorX_type, class VectorY_type, class VectorW_type>
int ComputeWAXPBY(const local_int_t n,
                  const typename VectorX_type::scalar_type alpha,
                  const VectorX_type & x,
                  const typename VectorY_type::scalar_type beta,
                  const VectorY_type & y,
                        VectorW_type & w,
                        bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  isOptimized = false;
  return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeWAXPBY< Vector<double>, Vector<double>, Vector<double> >(int, double, Vector<double> const&, double, Vector<double> const&, Vector<double>&, bool&);

template
int ComputeWAXPBY< Vector<float>, Vector<float>, Vector<float> >(int, float, Vector<float> const&, float, Vector<float> const&, Vector<float>&, bool&);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !KOKKOS_HALF_T_IS_FLOAT // if arch does not support half, then half = float
template
int ComputeWAXPBY< Vector<half_t>, Vector<half_t>, Vector<half_t> >(int, half_t, Vector<half_t> const&, half_t, Vector<half_t> const&, Vector<half_t>&, bool&);
#endif

// mixed
template
int ComputeWAXPBY< Vector<double>, Vector<float>, Vector<double> >(int, double, Vector<double> const&, float, Vector<float> const&, Vector<double>&, bool&);

#if defined(HPCG_WITH_KOKKOSKERNELS) & !KOKKOS_HALF_T_IS_FLOAT // if arch does not support half, then half = float
template
int ComputeWAXPBY< Vector<double>, Vector<half_t>, Vector<double> >(int, double, Vector<double> const&, half_t, Vector<half_t> const&, Vector<double>&, bool&);
#endif
