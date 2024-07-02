
#include "hip_global.h"

#define TPB 512

extern "C" {

__global__ void
__launch_bounds__(TPB, 1)
daxpy_kernel( int N, double a, double *x, double *y ){

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if ( tid >= N ) return;
  y[tid] = a * x[tid] + y[tid];  

}

void gpu_daxpy( int N, double a, double *x, double *y ){

  int n_blocks = (N - 1)/TPB + 1;
  daxpy_kernel<<<n_blocks, TPB>>>( N, a, x, y );
  
}


}