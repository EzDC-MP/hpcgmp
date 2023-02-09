
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

#include <fstream>

#include "hpgmp.hpp"
#include "DataTypes.hpp"

/*!
  Closes the I/O stream used for logging information throughout the HPCG run.

  @return returns 0 upon success and non-zero otherwise

  @see HPCG_Init
*/
int
HPCG_Finalize(void) {
  HPCG_fout.close();
#ifndef HPCG_NO_MPI
  #if defined(HPCG_WITH_KOKKOSKERNELS)
  MPI_Type_free(&HPGMP_MPI_HALF);
  MPI_Op_free(&MPI_SUM_HALF);
  #endif
#endif
  return 0;
}
