
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

#ifndef MYTIMER_HPP
#define MYTIMER_HPP
double mytimer(void);
void fence();

// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  fence(); t0 = mytimer()      //!< record current time in 't0'
#define TOCK(t) fence(); t += mytimer() - t0 //!< store time difference in 't' using time in 't0'


#endif // MYTIMER_HPP
