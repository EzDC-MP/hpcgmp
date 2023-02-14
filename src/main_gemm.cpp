
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"
#include "KokkosBlas2_gemv.hpp"

//using scalar_type = double;
//using scalar_type = float;
using scalar_type = Kokkos::Experimental::half_t;
using scalar_type2= float;

#if 1
// mixed-precision gemv
using half_t = Kokkos::Experimental::half_t;
using execution_space = typename Kokkos::DefaultExecutionSpace;
using memory_space    = typename execution_space::memory_space;
using AType  = Kokkos::View<scalar_type**, Kokkos::LayoutLeft, execution_space>;
using VType  = Kokkos::View<scalar_type* , Kokkos::LayoutLeft, execution_space>;

#include "KokkosBlas2_gemv_spec.hpp"
namespace KokkosBlas {
namespace Impl {
template struct GEMV<                                              \
    Kokkos::View<const half_t**, Kokkos::LayoutLeft,               \
                 Kokkos::Device<execution_space, memory_space>,    \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
    Kokkos::View<const half_t*, Kokkos::LayoutLeft,                \
                 Kokkos::Device<execution_space, memory_space>,    \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
    Kokkos::View<half_t*, Kokkos::LayoutLeft,                      \
                 Kokkos::Device<execution_space, memory_space>,    \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
    false, true>;
}
}
#endif

int main(int argc, char * argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int loops = 1;
    int calls = 1000;
    int warmup = 10;
    int M[3] = {1, 100, 1};
    int N[3] = {500000, 500000, 100};

    for(int i = 0; i < argc; i++) {
      if((strcmp(argv[i],"-loops")==0)) {
        loops = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i],"-M")==0)) {
        M[0] = atoi(argv[++i]);
        M[1] = atoi(argv[++i]);
        M[2] = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i],"-N")==0)) {
        N[0] = atoi(argv[++i]);
        N[1] = atoi(argv[++i]);
        N[2] = atoi(argv[++i]);
        continue;
      }
    }

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    {
      std::cout << std::endl;
      std::cout << "Default execution space: " << execution_space::name () << std::endl;
      std::cout << "Default memory space   : " << memory_space::name () << std::endl;
      std::cout << std::endl;
    }

#if defined(KOKKOS_HALF_T_IS_FLOAT)
  #if KOKKOS_HALF_T_IS_FLOAT
  std::cout << " > " <<sizeof(Kokkos::Experimental::half_t) << std::endl;
  #else
  std::cout << " + " <<sizeof(Kokkos::Experimental::half_t) << std::endl;
  #endif
#else
  std::cout << " x " <<sizeof(Kokkos::Experimental::half_t) << std::endl;
#endif

    const scalar_type  one  (1.0);
    const scalar_type2 zero (0.0);
    using  MType  = Kokkos::View<scalar_type**, Kokkos::LayoutLeft, execution_space>;
    using  VType  = Kokkos::View<scalar_type *, Kokkos::LayoutLeft, execution_space>;
    using  VType2 = Kokkos::View<scalar_type2*, Kokkos::LayoutLeft, execution_space>;

    Kokkos::Timer timer;
    printf( "\n" );
    printf( " M \t N \t time (s) \t Gflops\n" );
    printf( " ==================================================\n" );
    for (int m = M[0]; m <= M[1]; m+= M[2]) {
      for (int n = N[0]; n <= N[1]; n+= N[2]) {
        MType  A ("A", m, n);
        VType  x ("x", m);
        VType2 y ("y", n);
        Kokkos::Random_XorShift64_Pool<execution_space> random(13718);
        Kokkos::fill_random(A, random, scalar_type(1));
        Kokkos::deep_copy(x, one/scalar_type(m));
        Kokkos::deep_copy(y, zero);

        for (int ii = 0; ii < warmup; ii++) {
          KokkosBlas::gemv("T", one, A, x, zero, y);
        }
        Kokkos::fence();

        double enorm = 0.0;
        double gnorm = 0.0;
        {
          double gnorm_k = 0.0;
          auto A_host = Kokkos::create_mirror_view(A);
          auto x_host = Kokkos::create_mirror_view(x);
          auto y_host = Kokkos::create_mirror_view(y);
          Kokkos::deep_copy(y_host, y);
          for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
              gnorm_k += y_host(i) * y_host(i);
            }
          }
          gnorm += std::sqrt(gnorm_k);

          double enorm_k = 0.0;
          Kokkos::deep_copy(A_host, A);
          Kokkos::deep_copy(x_host, x);
          for (int i = 0; i < n; i++) {
            double e = 0.0; //y_host(i);
            for (int j = 0; j < m; j++) {
              e += (A_host(j,i) * x_host(j));
            }
            e = y_host(i) - e;
            enorm_k += e * e;
          }
          enorm += std::sqrt(enorm_k);
        }

        for (int nloop = 0; nloop < loops; nloop++) {
          timer.reset();
          for (int ii = 0; ii < calls; ii++) {
            KokkosBlas::gemv("T", one, A, x, zero, y);
          }
          Kokkos::fence();
          double gemm_time  = timer.seconds() / ((double)calls);
          double gemm_gflop = (2.0 * ((double)m * n))/(gemm_time*1000000000.0);
          printf( " %d \t %d \t %.5f \t %.2f \t   %.2e / %.2e = %.2e\n", m, n, gemm_time, gemm_gflop, enorm, gnorm, enorm / gnorm); fflush(stdout);
        }
      } // n
    } // m
    printf( "\n" );
  }
  Kokkos::finalize();
  return 0;
}
