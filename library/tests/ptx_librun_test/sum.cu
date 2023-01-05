#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define checkErrors(err) __cudaErrorCheck(err, __LINE__)
static inline void __cudaErrorCheck(CUresult err, const int line) {
  char *str;
  if (err != cudaSuccess) {
    cuGetErrorName(err, (const char **)&str);
    printf("[CUDA error] %04d \"%s\" line %d\n", err, str, line);
  }
}

__global__ void sum(int *x, int *y, int *z) {
  int tid = threadIdx.x;
  x[tid] += 1;
}

int main() {

	return 0;
}


