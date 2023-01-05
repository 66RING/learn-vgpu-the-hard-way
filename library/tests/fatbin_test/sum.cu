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

CUdevice device;
size_t totalGlobalMem;
CUcontext _context;
int block_size = 32;
CUfunction function;
char module_file[] = "sum.ptx";
char kernel_name[] = "_Z3sumPiS_S_";

void cudaRegisterFatbin() {
  // cuda driver API初始化
  cuInit(0);
  cuCtxCreate(&_context, 0, device);
}

void loadKernelFunction() {
  CUmodule module;
  CUresult err;

  // cuModuleLoad直接加载ptx文件
  //  其他api还要cuModuleLoadData等
  checkErrors(cuModuleLoad(&module, module_file));

  checkErrors(cuModuleGetFunction(&function, module, kernel_name));
}

int main() {
  // cuda初始化
  cudaRegisterFatbin();
  loadKernelFunction();

  int N = 32;
  int nbytes = N * sizeof(int);
  // int i = 0;
  int *dx = NULL, *hx = NULL;
  // 申请显存
  // cudaMalloc((void**)&dx, nbytes);
  checkErrors(cuMemAlloc((CUdeviceptr *)&dx, nbytes));

  // 申请成功
  if (dx == NULL) {
    printf("GPU alloc fail");
    return -1;
  }

  // 申请CPU内存
  hx = (int *)malloc(nbytes);
  if (hx == NULL) {
    printf("CPU alloc fail");
    return -1;
  }

  // init: hx: 0..31
  printf("hx original:\n");
  for (int i = 0; i < N; i++) {
    hx[i] = i;
    printf("%d ", hx[i]);
  }
  printf("\n");

  // copy to GPU
  // cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
  checkErrors(cuMemcpyHtoD((CUdeviceptr)dx, hx, nbytes));

  // call GPU
  // sum<<<1, N>>>(dx, dx ,dx);
  void **param = (void **)malloc(sizeof(void *) * 3);
  unsigned int sharedMemBytes = 0;
  CUstream hStream = 0;
  param[0] = &dx;
  param[1] = &dx;
  param[2] = &dx;
  checkErrors(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, sharedMemBytes, hStream, param, NULL));

  // wait gpu to finish
  cudaThreadSynchronize();

  // copy data to host
  // cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
  checkErrors(cuMemcpyDtoH(hx, (CUdeviceptr)dx, nbytes));

  printf("hx after:\n");
  for (int i = 0; i < N; i++) {
    printf("%d ", hx[i]);
  }

  printf("\n");
  cudaFree(dx);
  free(hx);
  return 0;
}


