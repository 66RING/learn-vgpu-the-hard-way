---
title: step3
author: 66RING
date: 2022-11-25
tags: 
- qemu
- gpu
mathjax: true
---

# 第三阶段: API实现细节

## 测试例程

数组每个元素加1, 原始数组0..31, 结果数组1..32

```c
#include<stdio.h>
#include<cuda.h>

typedef double FLOAT;
__global__ void sum(FLOAT *x) {
	int tid = threadIdx.x;
	x[tid] += 1;
}

int main() {
	int N = 32;
	int nbytes = N * sizeof(FLOAT);

	FLOAT *dx = NULL, *hx = NULL;
	int i;
	// 申请显存
	cudaMalloc((void**)&dx, nbytes);
	
	// 申请成功
	if (dx == NULL) {
		printf("GPU alloc fail");
		return -1;
	}

	// 申请CPU内存
	hx = (FLOAT*)malloc(nbytes);
	if (hx == NULL) {
		printf("CPU alloc fail");
		return -1;
	}

	// init: hx: 0..31
	printf("hx original:\n");
	for(int i=0;i<N;i++) {
		hx[i] = i;
		printf("%g\n", hx[i]);
	}

	// copy to GPU
	cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

	// call GPU
	sum<<<1, N>>>(dx);

	// let gpu finish
	cudaThreadSynchronize();

	// copy data to CPU
	cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);

	printf("hx after:\n");
	for(int i=0;i<N;i++) {
		printf("%g\n", hx[i]);
	}
	cudaFree(dx);
	free(hx);
	return 0;
}
```


## CUDA代码加载运行流程

顺序如下:

```
__cudaRegisterFatBinary
__cudaRegisterFunction
cudaMalloc
cudaMemcpy
cudaConfigureCall
cudaSetupArgument
cudaLaunch
cudaMemcpy
cudaFree
__cudaUnregisterFatBinary
```


```
# .gdbinit
b __cudaRegisterFunction
b __cudaUnregisterFatBinary
b __cudaRegisterFatBinary
b cudaMalloc
b cudaConfigureCall
b cudaLaunch
b cudaFree
b cudaSetupArgument
b cudaMemcpy
```


## CUDA API

### 设备初始化

TODO: 理清楚gpu context逻辑

- `cuDeviceGet`
- `cuCtxCreate`
- `cuCtxSetCurrent`
- `reloadAllKernels`


### __cudaRegisterFatBinary

> [what are the parameters for __cudaRegisterFatBinary and __cudaRegisterFunction functions?](https://stackoverflow.com/questions/6392407/what-are-the-parameters-for-cudaregisterfatbinary-and-cudaregisterfunction-f/39453201)

/usr/local/cuda/include/crt/host_runtime.h

`void** __cudaRegisterFatBinary(void *fatCubin);`

TODO: 为什么要注册, 因为要能table[]??

```
bin = __cudaRegisterFatBinary()
__cudaRegisterFunction(bin, funcName, deviceName)
TODO: 之后就可以使用cuModuleLoadData()等加载function了
```

`void *fatCubin`指向`struct __fatBinC_Wrapper_t`, 返回`fatCubin->data`作为handle供后续函数注册使用。

1. 前端返回`fatCubin->data`
2. 通知后端注册(初始化空表)
    1. `cuDeviceGetCount`获取设备数量, 初始化空的设备数据表
    2. 设备表可以用来记录context, function, 使用`cuDeviceGet`和`cuCtxCreate`创建设备
        a. TODO, 了解cuda device抽象


### __cudaRegisterFunction

```
void __cudaRegisterFunction(
        void   **fatCubinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize
);
```

TODO: review

1. 前端拿到handle指针后(本质是`fatBinaryHeader`)将header(和其他一些参数)交给后端
2. 后端为所有设备加载fatbin
    - 使用`cuModuleLoadData`和`cuModuleLoadData`直接加载header就可以(Q??: 内部会自动根据header找到数据)
3. `cudaStreamCreate()`
    - 默认使用stream 0
    - 创建stream数组, 后期`cudaStreamCreate`API使用


### cudaMalloc

`cudaError_t cudaMalloc ( void** devPtr, size_t size )`申请指定大小是设备内存, 保存到指针中

1. 直接转发, 结果返回


### cudaMemcpy

`cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )`在host和device之间拷贝数据, 传递方向由`kind`参数指定: 

- `cudaMemcpyHostToHost`
- `cudaMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost`
- `cudaMemcpyDeviceToDevice`
- `cudaMemcpyDefault`, 仅在支持uvm的gpu中适用

1. 判断传输方向
2. 从src拷贝count字节到dst

- API
    * TODO:

- TODO: 考虑mmap
    * 注意mmap的和非mmap的情况, 内存一致性


### cudaLaunch && cuLaunchKernel

`cudaError_t cudaLaunch (const void *func)`内部调用`cuLaunchKernel`完成功能

```c
CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra )
```

- f: Kernel to launch 
- gridDimX: Width of grid in blocks 
- gridDimY: Height of grid in blocks 
- gridDimZ: Depth of grid in blocks 
- blockDimX: X dimension of each thread block 
- blockDimY: Y dimension of each thread block 
- blockDimZ: Z dimension of each thread block 
- sharedMemBytes: Dynamic shared-memory size per thread block in bytes 
- hStream: Stream identifier 
- kernelParams: Array of pointers to kernel parameters 
- extra: Extra options

通过kernelParams给kernel设置参数的两种方式

1. N个参数通过`kernelParams`数组传输, **每个元素指向一个参数内存区域**, 区域大小不用指出, 因为kernel知道大小
2. TODO: 通过extra传输, 简易实现先不做

TODO: 前后端参数传输协议


1. 获取kernel配置信息
2. 获取kernel参数信息
    - 制作kernelParams: 每个元素都是指向参数的指针
3. 根据functionId在device表中找到对应kernel
4. TODO: stream, 什么是stream
5. cuLaunchKernel(func, stream, config, param, extra)启动


### cudaThreadSynchronize

`cudaError_t cudaThreadSynchronize (void)`

TODO:

直接转发, 等待设备完成


### cudaFree

`cudaError_t cudaFree ( void* devPtr )`

直接转发

### __cudaUnregisterFatBinary

```c
void __cudaUnregisterFatBinary(
  void **fatCubinHandle
);
```

TODO: 搞清楚为什么要register这种方式

- 调用`cuCtxDestroy`释放context
- `free`释放整设备表


### __cudaInitModule

```c
char __cudaInitModule(
        void **fatCubinHandle
);
```

暂时无用


## Ref

- ⭐⭐⭐ [CUDA API Remoting整理](https://juniorprincewang.github.io/2018/05/14/CUDA-API-Remoting/)
    * 内含很多链接
    * 有cudaRegisterFatBinary的内容
- [pause/resume](https://github.com/130B848/CRCUDA)
- [cuda wiki的整理](https://juniorprincewang.github.io/2019/07/31/cuda-wiki/#more)






