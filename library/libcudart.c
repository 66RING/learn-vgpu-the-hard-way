#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>

// from cuda header
#include <builtin_types.h>

#include<inttypes.h>
#include "../protocol/vgpu_common.h"

#define error(fmt, arg...) printf("ERROR: "fmt, ##arg)
#define panic(fmt, arg...) printf("panic at %s: "fmt, __FUNCTION__, ##arg); exit(-1)

const char dev_path[] = "/dev/vgpu";
int fd = -1;

static void device_open() {
	fd = open(dev_path, O_RDWR);
		if (fd < 0) {
		error("open device %s faild, %s (%d)\n", dev_path, strerror(errno), errno);
		exit(EXIT_FAILURE);
	}
}

static void send_to_device(VgpuArgs *args) {
	if (fd < 0)
		device_open();
	ioctl(fd, args->cmd, args);
}

// 申请内存资源, 并赋值到devPtr
cudaError_t cudaMalloc(void **devPtr, size_t size) {
	VgpuArgs args;
	memset(&args, 0, sizeof(VgpuArgs));

	args.cmd = VGPU_CUDA_MALLOC;
	// 获取线程id做为标识
	args.owner_id = syscall(__NR_gettid);
	args.dst_size = size;
	send_to_device(&args);
	*devPtr = (void*)args.dst;
	printf("cuda malloc 0x%x\n", (int)args.dst);
	return cudaSuccess;
}

// 释放GPU设备内存
cudaError_t cudaFree(void* devPtr) {
	VgpuArgs args;
	memset(&args, 0, sizeof(VgpuArgs));

	args.cmd = VGPU_CUDA_FREE;
	// 获取线程id做为标识
	args.owner_id = syscall(__NR_gettid);
	args.dst = (uint64_t)devPtr;
	printf("cuda free dst 0x%x\n", (int)args.dst);
	send_to_device(&args);
	return cudaSuccess;
}

// 记录gpu kernel配置参数, 调用cudaLaunchsh时将参数传入
cudaError_t cudaConfigureCall (dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
	// TODO:
	panic("unimplement");
	return cudaSuccess;
}

// 记录gpu kernel启动参数, 调用cudaLaunchsh时将参数传入
cudaError_t cudaSetupArgument (const void *arg, size_t size, size_t offset) {
	// TODO:
	panic("unimplement");
	return cudaSuccess;
}

// 执行cubin, 通过解析func可以获得到cubin的header
cudaError_t cudaLaunch (const void *func) {
	// TODO:
	panic("unimplement");
	return cudaSuccess;
}

// 解析fatCubin, 返回cubin指针
// 	涉及gpu ptx动态加载内容
void** __cudaRegisterFatBinary(void *fatCubin) {
	// TODO:
	panic("unimplement");
	return (void**)0;
}

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
) {
	panic("unimplement");
}
 

cudaError_t cudaThreadSynchronize (void) {
	panic("unimplement");
	return (cudaError_t)0;
}

void __cudaUnregisterFatBinary(
  void **fatCubinHandle
) {
	panic("unimplement");
}

char __cudaInitModule(
        void **fatCubinHandle
) {
	panic("unimplement");
	// TODO: 暂时没用
	return (char)0;
}
