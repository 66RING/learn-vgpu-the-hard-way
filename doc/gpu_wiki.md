---
title: GPU相关知识学习整理
author: 66RING
date: 2022-11-26
tags: 
- gpu
mathjax: true
---

# GPU wiki

## Terms

- fatbin
- ptx

## driver API

可以获取更多设备信息, 管理上下文, 管理module, 做显式初始化等

[cuda driver api](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)


## Context && Module

TODO:

context保存着使用的device信息(分配的mm, 加载的module, uvm map等), 但是context结果不公开

[详解CUDA的Context、Stream、Warp、SM、SP、Kernel、Block、Grid](https://zhuanlan.zhihu.com/p/266633373)


## stream

TODO

相当于GPU的流水线, 

```c
kernel<<<grid, block,Ns,stream>>>(param list);
```

- grid表示int型或者dim3类型（x,y,z)。用于定义一个grid中的block时如何组织的。int型则直接表示为1维组织结构
- block表示int型或者dim3类型（x,y,z)。用于定义一个block的thread是如何组织的。int型则直接表示为1维组织结构
- Ns 表示size_t类型，可缺省，默认为0.用于设置每个block除了静态分配的共享内存外，最多能动态分配的共享内存大小，单位为byte。0表示不需要动态分配
- stream表示cudaStream_t类型，可缺省，默认为0.表示该核函数位于哪个流


- ref
    * https://developer.download.nvidia.cn/CUDA/training/StreamsAndConcurrencyWebinar.pdf

## PTX

PTX(parallel thread execution)

英伟达提供了一些GPU专用的PTX指令用于gpu编程。用户可以内敛PTX汇编到cuda代码中以完成特定任务。

[ref](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#abstract)


## fatbin

⭐

涉及到如何将GPU kernel加载到gpu中。

将GPU kernel中二进制中分离出来

TODO: 二进制分离小实验验证 + gdb `b __cudaRegisterFatBinary`

`void** __cudaRegisterFatBinary(void *fatCubin);`接收一个`void*`参数, `fatCubin`指向`struct __fatBinC_Wrapper_t`结构, 里面含有cuda 二进制的信息和数据, 类似解析ELF。[__cudaRegisterFatBinary参数的讨论](https://stackoverflow.com/questions/6392407/what-are-the-parameters-for-cudaregisterfatbinary-and-cudaregisterfunction-f/39453201)

```c
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

`__fatBinC_Wrapper_t.data`指向具体的fatbin, 其内存布局是这样的`|struct fatBinaryHeader|data|`。即向通过`__fatBinC_Wrapper_t.data`解析出`fatBinaryHeader`, 然后用header指示的长度解析出后续数据。

```
// cuda9: /usr/local/cuda/include/fatbinary.h
struct __align__(8) fatBinaryHeader
{
	unsigned int 			magic;
	unsigned short         	version;
	unsigned short         	headerSize;
	unsigned long long int 	fatSize;
};
```

TODO:

- [Trouble launching CUDA kernels from static initialization code](https://stackoverflow.com/questions/24869167/trouble-launching-cuda-kernels-from-static-initialization-code/24883665#24883665)


## CUDA CUBIN/PTX文件动态加载

> 类型用户态协程: 创建上下文, 修改上下文, 加载上下文

存在一个可以动态加载模块`cubin.cu`内含若干函数。使用`-ptx`和`-arch=特定硬件`参数编译成`bin.cubin`

```c
// 使用extern "C"防止编译器改名
extern "C"   __global__  void kernel_run(){
    printf("hello world!\n");
}
```

如下流程(伪代码)可以实现动态加载。

```c
int main() {
    cuDeviceGet(&cuDevice, 0);
    // Creates a new CUDA context and associates it with the calling thread.
    cuCtxCreate(&cuContext, 0, cuDevice);
    // 加载模块
    cuModuleLoad (&module, "bin.cubin");
    // 加载模块中的函数
    cuModuleGetFunction(&mykernel,module,"kernel_run");
    // 执行
    cuLaunchKernel(mykernel, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
}
```


## ref

- [gpu wiki](https://juniorprincewang.github.io/2019/07/31/cuda-wiki/#more)
- [gpu wiki](https://github.com/yszheda/wiki/wiki/CUDA)
