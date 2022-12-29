---
title: 通过例子理解CUDA driver api和CUDA PTX
author: 66RING
date: 2022-12-28
tags: 
- cuda
- gpu
mathjax: true
---

# 通过例子理解CUDA driver api和CUDA PTX

cuda程序经过编译器编译后后添加很多对cuda driver api的调用, 这样用户就不用关心gpu module管理, context管理, kernel管理等的细节了。

不过我们就是想要知道细节, 所以这篇文章通过直接使用cuda driver api加载cuda ptx执行的方式体会其中的细节。

我们的目标是将下面这个cuda程序转换成"手动挡"。这个cuda程序的功能是对数组的每个元素加1。

```c
// sum.cu
#include <cuda.h>
#include <stdio.h>

__global__ void sum(int *x, int *y, int *z) {
  int tid = threadIdx.x;
  x[tid] += 1;
}

int main() {
  int N = 32;
  int nbytes = N * sizeof(int);
  int *dx = NULL, *hx = NULL;
  // 申请显存
  cudaMalloc((void **)&dx, nbytes);

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
  cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

  // call GPU
  sum<<<1, N>>>(dx, dx, dx);

  // let gpu finish
  cudaThreadSynchronize();

  // copy data to CPU
  cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);

  printf("hx after:\n");
  for (int i = 0; i < N; i++) {
    printf("%d ", hx[i]);
  }
  printf("\n");
  cudaFree(dx);
  free(hx);
  return 0;
}
```


## 基本流程

> 本实验基于CUDA 9.0, ubuntu18.04
>
> 简单的说cuda kernel就是这东西: `kernel<<<gridDim, blockDim,sharedMemorySize,stream>>>(param, ...);`

GPU作为一种计算件(具有计算能力的设备)它应该和CPU是差不多的(除了图灵奖/asm还能有啥)，都需要加载二进制然后一路pc++。

想要知道内部流程最好的方法就是让操作系统告诉我们：

1. 先使用动态链接的方式编译cuda程序, 这样他会为我们过滤掉其他没有使用的api: `$ nvcc --cudart shared -o sum sum.cu`
2. 使用`nm`工具查看二进制都用到了哪些符号
```
$ nm ./sum | grep libcudart
                 U cudaConfigureCall@@libcudart.so.9.0
                 U cudaFree@@libcudart.so.9.0
                 U __cudaInitModule@@libcudart.so.9.0
                 U cudaLaunch@@libcudart.so.9.0
                 U cudaMalloc@@libcudart.so.9.0
                 U cudaMemcpy@@libcudart.so.9.0
                 U __cudaRegisterFatBinary@@libcudart.so.9.0
                 U __cudaRegisterFunction@@libcudart.so.9.0
                 U cudaSetupArgument@@libcudart.so.9.0
                 U cudaThreadSynchronize@@libcudart.so.9.0
                 U __cudaUnregisterFatBinary@@libcudart.so.9.0
```
3. gdb打断点确认执行顺序
    - 可能用nv的库看不到符号表信息, 我这里用的gpgpu-sim模拟的
    - 还有一种方法就是用`LD_PRELOAD`挨个测试

可以发现调用顺序如下

```
__cudaRegisterFatBinary     // 加载cuda二进制, 就相当于exec, 只不过是给gpu加载
__cudaRegisterFunction      // 具体函数的加载, 一个cuda程序可能存在多个kernel, cuda kernel又有若干参数<<<>>>
cudaMalloc                  // cuda申请内存
cudaMemcpy                  // 数据在gpu内和系统内存直接拷贝
cudaConfigureCall           // 准备kernel调用的配置信息, 如几个grid, block, 哪个stream等
cudaSetupArgument           // 准备kernel调用的参数, 将函数参数压到参数栈中, 有几个参数调用几次
cudaLaunch                  // 执行kernel调用
cudaMemcpy                  // 执行完成数据拷贝会系统内存
cudaFree                    // 释放gpu内存资源
__cudaUnregisterFatBinary   // cuda程序执行完成, 计算资源从gpu中撤离
```

可以看到gpu相当于一个异步运行线程, 执行一个cuda程序前要先注册可能的调用, 执行完成后要卸载任务。

一个kernel调用`kernel<<<grid, block,Ns,stream>>>(param list);`会被分解成三个调用:

1. cudaConfigureCall
2. cudaSetupArgument
3. cudaLaunch

总结一下kernel的加载执行的步骤就是(其实和操作系统加载解析ELF差不多): 

1. 创建上下文, 加载二进制文件, 解析出要调用的kernel
2. 资源初始化: host to device, device to host
3. kernel启动(调用)

经过一番查阅文档后, 上述步骤对应成代码就是:

### 第一步

```c
// 1. 创建上下文, 加载二进制文件, 解析出要调用的kernel
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
```

几个注意个点

- `cuInit(0)`用于初始化cuda driver api, 使用driver api前调用一下
- 如何获取ptx文件?
    * cuda程序编译是添加`-keep`参数保留中间结果
    * e.g. `nvcc -keep -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -I/usr/local/cuda/include ./sum.cu -o sum -lcuda -lcudart`
- 通过kernel name加载, 怎么知道name是什么?
    * 使用`objdump -d ./sum`反汇编查看二进制的符号信息, 函数前面经过编译后稍微有点变化的
    ```
    $ objdump -d ./sum
    ...
    00000000004015ee <_Z3sumPdS_S_>:
      4015ee:       55                      push   %rbp
      4015ef:       48 89 e5                mov    %rsp,%rbp
    ...
    ```
- 最后会返回当前上下文的一个function handle, 我们先保存起来后续要用来调用执行

### 第二步

资源申请和初始化等也换成driver API

```c
// 申请显存
// cudaMalloc((void**)&dx, nbytes);
checkErrors(cuMemAlloc((CUdeviceptr *)&dx, nbytes));

// host拷贝到gpu
// cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
checkErrors(cuMemcpyHtoD((CUdeviceptr)dx, hx, nbytes));

// gpu拷贝到host
// cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
checkErrors(cuMemcpyDtoH(hx, (CUdeviceptr)dx, nbytes));
```

资源拷贝中不同方向的driver API不同, 这里有device到host的`cuMemcpyDtoH`和host到device`cuMemcpyHtoD`。同理还要DtoD, HtoH等。


### 第三步

kernel调用。cuda kernel是这样一个东西: `kernel<<<gridDim, blockDim,sharedMemorySize,stream>>>(param, ...);`编译器会解析出这个语句的kernel config, 即gridDim等, 然后转换成c语言的函数调用。具体调用的就是`cuLaunchKernel`。

这里的话需要看一下[官方文档](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)。可以看到通过`void** kernelParams`传递参数数组, 而参数的类型不同长度不同, 这些具体地的怎么传递的呢？

方法一：有N个参数, kernelParams是一个长度为N的数组, 数组的每个元素都散指向参数存储位置的指针。对于参数指针的大小我们不需要特别指定, 因为类型的大小信息已经被编译到kernel中了

方法二: 通过extra参数传递, 用于一些小众的场景。传递到extra的是一个name, value数组。name后就紧跟value, 如此往复, 遇到NULL或`CU_LAUNCH_PARAM_END`停止。例如

```c
size_t argBufferSize;
      char argBuffer[256];
  
      // populate argBuffer and argBufferSize
  
      void *config[] = {
          CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
          CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
          CU_LAUNCH_PARAM_END
      };
      status = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, s, NULL, config);
```

- `CU_LAUNCH_PARAM_BUFFER_POINTER`和它后面的value指示一个包含所有参数的buffer
- `CU_LAUNCH_PARAM_BUFFER_SIZE`和后面的value知识buffer的大小

其实和第一种方式的一样的。所以这里就用第一种

```c
  // call GPU
  // sum<<<1, N>>>(dx, dx ,dx);
  unsigned int sharedMemBytes = 0;
  CUstream hStream = 0;
  // 准备参数数组, 每个元素都散指向参数的指针
  void **param = (void **)malloc(sizeof(void *) * 3);
  param[0] = &dx;
  param[1] = &dx;
  param[2] = &dx;
  checkErrors(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, sharedMemBytes, hStream, param, NULL));

  // wait gpu to finish
  cudaThreadSynchronize();
```


## 最终代码

```c
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
```

实验一下: 确实每个元素加一了

```
$ nvcc -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -I/usr/local/cuda/include ./sum.cu -o sum -lcuda -lcudart
$ ./sum
hx original:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
hx after:
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
```

> 支持一波噻: [博客](66ring.github.io/), [github](https://github.com/66RING)
