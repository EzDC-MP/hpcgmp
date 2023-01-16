
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

#define START_T()  fence(); start_t = mytimer()      //!< record current time in 'start_t'
#define STOP_T(t)  fence(); t += mytimer() - start_t //!< store time difference in 't' using time in 'start_t'


#endif // MYTIMER_HPP
