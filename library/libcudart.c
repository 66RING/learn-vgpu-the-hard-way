#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <assert.h>

// cuda header
#include <builtin_types.h>
// cuda 9.0 only
#include <fatBinaryCtl.h> // struct __fatBinC_Wrapper_t
#include <fatbinary.h> // struct fatBinaryHeader

#include<inttypes.h>
#include "../protocol/vgpu_common.h"

// cuda9: /usr/local/cuda/include/fatbinary.h
// struct __align__(8) fatBinaryHeader
// {
// 	unsigned int 			magic;
// 	unsigned short         	version;
// 	unsigned short         	headerSize;
// 	unsigned long long int 	fatSize;
// };

// cuda9: /usr/local/cuda/include/fatBinaryCtl.h
// typedef struct {
// 	int magic;
// 	int version;
// 	const unsigned long long* data;
// 	void *filename_or_fatbins;  /* version 1: offline filename,
//                                * version 2: array of prelinked fatbins */
// } __fatBinC_Wrapper_t;
// #define FATBIN_MAGIC 0x466243b1


#define error(fmt, arg...) printf("ERROR: "fmt, ##arg)
#define panic(fmt, arg...) printf("panic at %s: line %d"fmt, __FUNCTION__, __LINE__, ##arg); exit(-1)

#if 1
	#define dprintf(fmt, arg...) printf("DEBUG: "fmt, ##arg)
#else
	#define dprintf(fmt, arg...)
#endif


// 记录内核配置
uint64_t cudaKernelConf[8];
// 记录若干个参数内核启动参数
#define cudaKernelParaStackMaxSize 512
// 我们传递cudaKernelPara这片连续空间到内核态
//  因为我们要传递的数据大小就是sizeof(uint32_t) + cudaKernelPara.paraStackSize
struct {
	// 指示参数数量
	uint32_t paraNum;
	// 保存参数数据: (参数类型长度(uint32_t), 参数数据)
	uint8_t paraStack[cudaKernelParaStackMaxSize];
	// (sub header, data) + ... 的总长度
	uint32_t paraStackOffset;
} cudaKernelPara;


const char dev_path[] = "/dev/vgpu";
int fd = -1;

static void device_open() {
	fd = open(dev_path, O_RDWR);
		if (fd < 0) {
		error("open device %s faild, %s (%d)\n", dev_path, strerror(errno), errno);
		exit(EXIT_FAILURE);
	}
}

static void send_to_driver(VgpuArgs *args) {
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
	send_to_driver(&args);
	*devPtr = (void*)args.dst;
	dprintf("cuda malloc 0x%lx\n", args.dst);
	return args.ret;
}

// 释放GPU设备内存
cudaError_t cudaFree(void* devPtr) {
	VgpuArgs args;
	memset(&args, 0, sizeof(VgpuArgs));

	args.cmd = VGPU_CUDA_FREE;
	// 获取线程id做为标识
	args.owner_id = syscall(__NR_gettid);
	args.dst = (uint64_t)devPtr;
	dprintf("cuda free dst 0x%lx\n", args.dst);
	send_to_driver(&args);
	return args.ret;
}

// count: count in byte
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
	VgpuArgs args;
	memset(&args, 0, sizeof(VgpuArgs));

	// 获取线程id做为标识
	args.owner_id = syscall(__NR_gettid);

	args.cmd = VGPU_CUDA_MEMCPY;
	args.dst = (uint64_t)dst;
	args.src = (uint64_t)src;
	// size一次只会用到一个, 直接设置整相同, 不需要额外判断
	args.src_size = (uint64_t)count;
	args.dst_size = (uint64_t)count;
	args.kind = (int)kind;
	dprintf("kind %d\n", kind);
	send_to_driver(&args);
	// TODO: error handling
	return args.ret;
}


// 记录gpu kernel配置参数, 调用cudaLaunchsh时将参数传入
cudaError_t cudaConfigureCall (dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    dprintf("gridDim: %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
    dprintf("blockDim: %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
    dprintf("sharedMem: %lu\n", sharedMem);
    dprintf("stream: %p\n", (void *) stream);

    cudaKernelConf[0] = gridDim.x;
    cudaKernelConf[1] = gridDim.y;
    cudaKernelConf[2] = gridDim.z;

    cudaKernelConf[3] = blockDim.x;
    cudaKernelConf[4] = blockDim.y;
    cudaKernelConf[5] = blockDim.z;

    cudaKernelConf[6] = sharedMem;

    cudaKernelConf[7] = (stream == NULL) ? (uint64_t) 0 : (uint64_t) stream;

	// 参数数量清零
	cudaKernelPara.paraNum = 0;
	// 数据位置偏移清零
	cudaKernelPara.paraStackOffset = 0;
	return cudaSuccess;
}

// 记录gpu kernel启动参数, 调用cudaLaunchsh时将参数传入
//  将参数push到一个区域中(参数栈), offset就是当前参数的byte偏移量
//  有几个参数就会调用几次
cudaError_t cudaSetupArgument (const void *arg, size_t size, size_t offset) {
	// 参数类型长度(uint32_t) + 参数数据大小
	memcpy(&cudaKernelPara.paraStack[cudaKernelPara.paraStackOffset], &size, sizeof(uint32_t));
	dprintf("param size: 0x%x\n", *(uint32_t*)&cudaKernelPara.paraStack[cudaKernelPara.paraStackOffset]);
	cudaKernelPara.paraStackOffset += sizeof(uint32_t);

	memcpy(&cudaKernelPara.paraStack[cudaKernelPara.paraStackOffset], arg, size);
	dprintf("param value: %llx\n", *(unsigned long long *) &cudaKernelPara.paraStack[cudaKernelPara.paraStackOffset]);
	cudaKernelPara.paraStackOffset += size;

	// 参数数量++
	cudaKernelPara.paraNum ++;
	return cudaSuccess;
}

// 执行cubin, 通过解析func可以获得到cubin的header
cudaError_t cudaLaunch (const void *func) {
	// 保证内存空间连续
	assert((uint64_t)&cudaKernelPara.paraStack == sizeof(uint32_t) + (uint64_t)&cudaKernelPara);


	// TODO:
	panic("unimplement");
	return cudaSuccess;
}

// 解析fatCubin, 返回cubin指针
// 	涉及gpu ptx动态加载内容
void** __cudaRegisterFatBinary(void *fatCubin) {
	// TODO:
	unsigned int magic;
	void **fatCubinHandle;
	magic = *(unsigned int *) fatCubin;
	// fatBinaryCtl.h
    if (magic != FATBINC_MAGIC) {
		panic("unknown cuda magic 0x%x, expect 0x%x\n", magic, FATBINC_MAGIC);
	}

	fatCubinHandle = malloc(sizeof(void *)); //original

	__fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t *) fatCubin;
	// TODO: 为何需要承载?
	// 不要行不行?
	*fatCubinHandle = (void*)binary->data;
	dprintf("cuda register fatCubin: 0x%lx\n", (uint64_t)fatCubin);
	dprintf("magic: %x\n", binary->magic);
	dprintf("version: %x\n", binary->version);
	dprintf("data: %p\n", binary->data);
	dprintf("filename_or_fatbins: %p\n", binary->filename_or_fatbins);


	VgpuArgs args;
	memset(&args, 0, sizeof(VgpuArgs));
	args.cmd = VGPU_CUDA_REGISTER_FAT_BINARY;
	// 仅用于通知后端初始化设备和上下文
	send_to_driver(&args);
	
	// TODO: handle 只是一个标识
	// *fatCubinHandle = data
	return fatCubinHandle;
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

    dprintf("fatCubinHandle: %p, value: %p\n", fatCubinHandle, *fatCubinHandle);
    dprintf("hostFun: %s (%p)\n", hostFun, hostFun);
    dprintf("deviceFun: %s (%p)\n", deviceFun, deviceFun);
    dprintf("deviceName: %s\n", deviceName);
    dprintf("thread_limit: %d\n", thread_limit);

	VgpuArgs args;
	memset(&args, 0, sizeof(VgpuArgs));

	computeFatBinaryFormat_t fatBinHeader;
	fatBinHeader = (computeFatBinaryFormat_t) (*fatCubinHandle);

	// 获取线程id做为标识
	args.owner_id = syscall(__NR_gettid);

	args.cmd = VGPU_CUDA_REGISTER_FUNCTION;
	args.src = (uint64_t)(fatBinHeader);
	args.src_size = (uint64_t)fatBinHeader->fatSize;
	args.dst = (uint64_t)deviceName;
	args.dst_size = (uint64_t)strlen(deviceName) + 1;
	args.flag = (uint32_t) (uint64_t) hostFun;

	send_to_driver(&args);
	// TODO: error handling
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
	// TODO: 暂时没用
	return (char)0;
}
