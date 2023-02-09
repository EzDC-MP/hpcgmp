
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
 @file ExchangeHalo.cpp

 HPCG routine
 */

// Compile this routine only if running with MPI
#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "Utils_MPI.hpp"
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include <cstdlib>

/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */
template<class SparseMatrix_type, class Vector_type>
void ExchangeHalo(const SparseMatrix_type & A, Vector_type & x) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType ();

  // Extract Matrix pieces
  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t * receiveLength = A.receiveLength;
  local_int_t * sendLength = A.sendLength;
  int * neighbors = A.neighbors;
  scalar_type * sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * elementsToSend = A.elementsToSend;

  scalar_type * const xv = x.values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(A.comm, &size);
  MPI_Comm_rank(A.comm, &rank);
  if (size == 1) return;

  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  int MPI_MY_TAG = 99;

  MPI_Request * request = new MPI_Request[num_neighbors];

  //
  // Externals are at end of locals
  //
  scalar_type * x_external = (scalar_type *) xv + localNumberOfRows;

  // Post receives first
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, A.comm, request+i);
    x_external += n_recv;
  }


  //
  // Fill up send buffer
  //

  // TODO: Thread this loop
  for (local_int_t i=0; i<totalToBeSent; i++) sendBuffer[i] = xv[elementsToSend[i]];

  //
  // Send to each neighbor
  //

  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    MPI_Send(sendBuffer, n_send, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, A.comm);
    sendBuffer += n_send;
  }

  //
  // Complete the reads issued above
  //

  MPI_Status status;
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    if ( MPI_Wait(request+i, &status) ) {
      std::exit(-1); // TODO: have better error exit
    }
  }

  delete [] request;

  return;
}


/* --------------- *
 * specializations *
 * --------------- */

template
void ExchangeHalo< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double>&);

template
void ExchangeHalo< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float>&);

#if defined(HPCG_WITH_KOKKOSKERNELS)
template
void ExchangeHalo< SparseMatrix<half_t>, Vector<half_t> >(SparseMatrix<half_t> const&, Vector<half_t>&);
#endif

#endif // ifndef HPCG_NO_MPI
