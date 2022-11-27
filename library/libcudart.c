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

static void send_command(VgpuArgs *args) {
	if (fd < 0)
		device_open();
	ioctl(fd, args->cmd, args);
}

// 申请内存资源, 并赋值到devPtr
cudaError_t cudaMalloc(void **devPtr, size_t size) {
	VgpuArgs args;
	memset(&args, 0, sizeof(VgpuArgs));

	args.cmd = VGPU_CUDA_MALLOC;
	args.owner_id = syscall(__NR_gettid);
	// TODO:
	send_command(&args);
	*devPtr = (void*)args.dst;
	printf("0x%x\n", (int)args.dst);
	return (cudaError_t)0;
}

cudaError_t cudaConfigureCall (dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
	// TODO:
	panic("unimplement");
	return (cudaError_t)0;
}

cudaError_t cudaSetupArgument (const void *arg, size_t size, size_t offset) {
	// TODO:
	panic("unimplement");
	return (cudaError_t)0;
}

cudaError_t cudaLaunch (const void *func) {
	// TODO:
	panic("unimplement");
	return (cudaError_t)0;
}

void** __cudaRegisterFatBinary(void *fatCubin) {
	// TODO:
	panic("unimplement");
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
