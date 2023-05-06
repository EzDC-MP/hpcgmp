
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Ichitaro Yamazaki         (iyamaza@sandia.gov)
// Sivasankaran Rajamanickam (srajama@sandia.gov)
// Piotr Luszczek            (luszczek@eecs.utk.edu)
// Jack Dongarra             (dongarra@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef GENERATEGEOMETRY_HPP
#define GENERATEGEOMETRY_HPP
#include "Geometry.hpp"
void GenerateGeometry(int size, int rank, int numThreads, int pz, local_int_t zl, local_int_t zu, local_int_t nx, local_int_t ny, local_int_t nz, int npx, int npy, int npz, Geometry * geom);
#endif // GENERATEGEOMETRY_HPP
