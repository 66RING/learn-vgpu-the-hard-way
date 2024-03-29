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


可以发现调用顺序如下

```
__cudaRegisterFatBinary     // 加载cuda二进制, 就相当于exec, 只不过是给gpu加载
__cudaRegisterFunction      // 具体函数的加载, 一个cuda程序可能存在多个kernel, cuda kernel又有若干参数<<<>>>
cudaMalloc                  // cuda申请内存
cudaMemcpy                  // 数据在gpu内和系统内存直接拷贝
cudaConfigureCall           // 准备kernel调用的配置信息, 如几个grid, block, 哪个stream等
cudaSetupArgument           // 准备kernel调用的参数, 也就是函数参数
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


## CUDA API

### 设备初始化

TODO: 理清楚gpu context逻辑。但是gpgpu-sim环境下部分API没有提供, 故先忽略

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

`void* fatCubin`指向的结构如下, 其中`const unsigned long long* data`指向fatbin的header, cuda driver api可以通过这段header区域加载image/module, 然后解析加载kernel

```cpp
// cuda9: /usr/local/cuda/include/fatBinaryCtl.h
#define FATBINC_MAGIC   0x466243B1
#define FATBINC_VERSION 1
#define FATBINC_LINK_VERSION 2
typedef struct {
	int magic;
	int version;
	const unsigned long long* data;
	void *filename_or_fatbins;  /* version 1: offline filename,
                               * version 2: array of prelinked fatbins */
} __fatBinC_Wrapper_t;
```

下面这个结构即是要用来加载的header, **整个image的长度为headerSize + fatSize**

```cpp
// cuda9: /usr/local/cuda/include/fatbinary.h
struct __align__(8) fatBinaryHeader
{
	unsigned int 			magic;
	unsigned short         	version;
	unsigned short         	headerSize;
	unsigned long long int 	fatSize;
};
```


### __cudaRegisterFunction

```cpp
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

官方版的`void** fatCubinHandle`就是指向的就是`__cudaRegisterFatBinary(void *fatCubin)`中的`*fatCubin`, 即`*fatCubinHandle == fatCubin`, 即`fatCubinHandle`数组中的元素是一个指针。


1. 前端拿到handle指针后(本质是`fatBinaryHeader`)将header(和其他一些参数)交给后端
    - 最终会需要fatBin: 数据载体, fucntionName: 从fatBin中加载函数, hostFun: 作为funcId标识
2. 后端为所有设备加载fatbin
    - 使用`cuModuleLoadData`和`cuModuleLoadData`直接加载header就可以(Q??: 内部会自动根据header找到数据)
3. `cudaStreamCreate()`
    - 默认使用stream 0
    - 创建stream数组, 后期`cudaStreamCreate`API使用

TODO

- 一个device可以有多个function
- TODO: 总结需要的map
- module management
    * https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDA__MODULE_ga52be009b0d4045811b30c965e1cb2cf.html#ga52be009b0d4045811b30c965e1cb2cf

context = module管理 + kernel管理 + ...

- image加载到上下文, 以module形式返回, 函数从module中加载，最后暴露给用户function handle

- thread id -> 找到对应device 
- thread id -> context
    * 执行前加载context
    * TODO: 先把一次性实现


### cudaMalloc

`cudaError_t cudaMalloc ( void** devPtr, size_t size )`申请指定大小是设备内存, 保存到指针中

1. 直接转发, 结果返回


### cudaMemcpy

> ⭐

`cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )`在host和device之间拷贝数据, 传递方向由`kind`参数指定: 

- `cudaMemcpyHostToHost`
- `cudaMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost`
- `cudaMemcpyDeviceToDevice`
- `cudaMemcpyDefault`, 仅在支持uvm的gpu中适用

1. 判断传输方向
2. 从src拷贝count字节到dst


- API
    * TODO: 补全API功能说明
    * cuda driver API
        + cuMemcpyHtoD, ... 系列
    * kernel driver API
        + `virt_to_phys()`
            + 内核地址转物理地址
        + `copy_from_user`
    * qemu API
        + `memory_region_find()`
        + `get_system_memory()`
        + `memory_region_is_ram()`
        + `memory_region_get_ram_ptr()`
        + `qemu_map_ram_ptr()`

#### 设备初始化

- `cuInit(0)`: 初始化cuda driver API


#### Host to Device

- src: 主存内存地址, 需要user, kernel, gpa, hva的转换
- dst: 设备内存地址, 直接传递, 不需要转换

- driver
    1. 用户态数据拷贝: `copy_from_user`
    2. 内核空间地址转换, 转换成VMM识别的gpa(linux内核地址 != gpa): `virt_to_phys`
    3. 发送命令
    4. 释放临时buffer
        - 注意此时src是经过`virt_to_phys`转换后的物理地址, 需要`phys_to_virt`后才能kfree
- backend
    1. 虚拟机内存地址转换
        - `memory_region_find`找memory region section 
        - `memory_region_get_ram_ptr`通过section的mr找对应的hva区间(ram region) 
        - 计算section在ram region中的偏移
    2. 调用cuda driver API完成cudaMemcpy
    3. TODO: mmap带来的一致性问题


#### Device to Host

- src: 设备内存地址, 直接传递, 不需要转换
- dst: 主存内存地址, 需要user, kernel, gpa, hva的转换

- driver
    1. 开辟内核态缓存用于接收数据
    2. 缓存转换gpa告知后端, 用于接收数据
    3. 发送命令
    4. 数据从内核态拷贝到用户态`copy_to_user()`, 注意gpa将转换到virt
- backend
    1. 获取目的地址hva
    2. 数据拷贝到hva: `cuMemcpyDtoH`, 此时内核态buffer就被接收到了数据


####  难点

- result
    * linux内核地址 != 物理地址, 即不是需要的gpa, 还需要一次`virt_to_phys()`转换
    * 注意传输的单位是byte
    * ⭐ 需要device初始化


- 考虑虚拟机内部内存地址转换问题
- 考虑driver处理用户空间地址转换
- 考虑mmap时设备内存和host内存混用的一致性问题
    * TODO: 先不考虑mmap的情况

- TODO: 考虑mmap
    * 注意mmap的和非mmap的情况, 内存一致性


### cudaConfigureCall

```c
cudaError_t cudaConfigureCall(
        dim3 gridDim,
        dim3 blockDim,
        size_t sharedMem,
        cudaStream_t stream);
```

1. 前端保存用户配置
2. 待`cudaLaunch`时发送给后端


### ⭐ cudaSetupArgument

```c
cudaError_t cudaSetupArgument (const void *arg, size_t size, size_t offset);

// Pushes size bytes of the argument pointed to by arg at offset bytes from the start
// of the parameter passing area, which starts at offset 0. The arguments are stored in
// the top of the execution stack. cudaSetupArgument() must be preceded by a call to
// cudaConfigureCall().
```

**注意是push,  有几个参数就会调用几次**, 将参数push到一个区域中, offset就是当前参数的byte偏移

- arg: Argument to push for a kernel launch
- size Size of argument
- offset: Offset in argument stack to push new arg
    * 因为有几个参数就会调用几次, 所以这个offset就当前参数的位置偏移, 单位是byte

1. 前端保存内核启动参数
2. 考虑变长参数情况

创建一个parameter passing area(数组 + size), 然后看后面`cuLaunchKernel`是怎么使用它的。

因为我们在`cuLaunchKernel`往内核态传的时候需要传递这个变长的数据, 而传递变长数据的方法往往是在头部添加一个header。我们这里需要**考虑两种变长的情况**

- 参数数量的变长
- 参数类型的变长

所以我们为了方便就不能老实遵守它这套机制了, 因为我们的后端并不知道这些变长的信息, 我们通过如下结构传递

```c
// 我们传递cudaKernelPara这片连续空间到内核态
//  因为我们要传递的数据大小就是sizeof(uint32_t) + cudaKernelPara.paraStackOffset
struct {
	// 指示参数数量
	uint32_t paraNum;
	// 保存参数数据: (参数类型长度(uint32_t), 参数数据)
	uint8_t paraStack[cudaKernelParaStackMaxSize];
	// (sub header, data) + ... 的总长度
	uint32_t paraStackOffset;
} cudaKernelPara;
```

然后我们在`cudaLaunch`中加个断言`(uint64_t)&cudaKernelPara.paraStack == sizeof(uint32_t) + (uint64_t)&cudaKernelPara`保证内存布局连续。这个全局信息在`cudaConfigureCall`中初始化


### cudaLaunch && cuLaunchKernel

`cudaError_t cudaLaunch (const void *func)`内部调用`cuLaunchKernel`完成功能

```c
CUresult cuLaunchKernel ( CUfunction f, 
    unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
    unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
    unsigned int  sharedMemBytes, CUstream hStream, 
    void** kernelParams, void** extra )
```

- f: Kernel to launch, fuction handle
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

- 前端
    * 从用户态打包到内核态连续空间, 这样才能以连续物理地址返回
    * 而这段连续空间怎么打包的就需要特定的协议了, 即encode, decode, 因为
        + 内核态并没有参数类型信息, 不知道参数的数量
    * 协议: header, 
- 后端
    * 拿到前端数据后根据CUDA driver API文档组织参数形式然后传递给cuLaunchKernel

- TODO: 整理
    * function handle -> kernel


#### 参数传递的方式

可以看到通过`void** kernelParams`传递参数数组, 而参数的类型不同长度不同, 这些具体地的怎么传递的呢？

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

其实和第一种方式的一样的

[官方文档](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)


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


## CUDA driver API

- `cuInit(0)`, 初始化cuda driver api, 在任何driver api前使用, 会因为driver不匹配失败
- `cuDeviceGet(&device, 0)`获取第一个设备
- `cuCtxCreate(&context, 0, device)`, 创建context


```c
CUresult loadKernelFunction()
{
    // 这里的module_file是nvcc将kernel code编译成的ptx文件，这里用的是offline static compilation。
    // 也可以使用nvrtc实现online comilation。产生后的PTX代码，使用cuModuleLoadData加载module，使用cuLinkAddData进行link。
    // 也可以通过cuModuleLoadFatBinary直接导入fatbin文件 
    err = cuModuleLoad(&module, module_file);
    err = cuModuleGetFunction(&function, module, kernel_name);
    return err;
}
```


## Ref

- ⭐⭐⭐ [CUDA API Remoting整理](https://juniorprincewang.github.io/2018/05/14/CUDA-API-Remoting/)
    * 内含很多链接
    * 有cudaRegisterFatBinary的内容
- [pause/resume](https://github.com/130B848/CRCUDA)
- [cuda wiki的整理](https://juniorprincewang.github.io/2019/07/31/cuda-wiki/#more)







