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

## stream

TODO


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
