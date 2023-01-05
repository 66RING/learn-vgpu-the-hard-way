#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <assert.h>
#include <dlfcn.h>
#include <stdlib.h>

// cuda header
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_profiler_api.h>

// cuda 9.0 only
#include <fatBinaryCtl.h> // struct __fatBinC_Wrapper_t
#include <fatbinary.h> // struct fatBinaryHeader

#include<inttypes.h>

// cuda9: /usr/local/cuda/include/fatbinary.h
// typedef struct fatBinaryHeader* computeFatBinaryFormat_t
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

#define cudaErrorCheck(err) __cudaErrorCheck(err, __LINE__)
static inline void __cudaErrorCheck(CUresult err, const int line) {
    char *str;
    if (err != CUDA_SUCCESS) {
		cuGetErrorName(err, (const char **) &str);
        printf("[CUDA error] %04d \"%s\" line %d\n", err, str, line);
    }
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

void loadKernelFunction(const void* image) {
  CUmodule module;

  // // cuModuleLoad直接加载ptx文件
  // //  其他api还要cuModuleLoadData等
  // cudaErrorCheck(cuModuleLoad(&module, module_file));

  // cudaErrorCheck(cuModuleGetFunction(&function, module, kernel_name));


	// 从fatbin中加载module到当前context, 返回module
	cudaErrorCheck(cuModuleLoadData(&module, image));

	// 从module中加载函数, 返回一个funcHandle
    cudaErrorCheck(cuModuleGetFunction(&function, module, kernel_name));

}


// 记录内核配置
uint64_t cudaKernelConf[8];
// 记录若干个参数内核启动参数
#define cudaKernelParaStackMaxSize 512
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

	fatCubinHandle = malloc(sizeof(void *));

	__fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t *) fatCubin;
	// TODO: 为何需要承载?
	// 不要行不行?
	*fatCubinHandle = (void*)binary->data;
	dprintf("cuda register fatCubin: 0x%lx\n", (uint64_t)fatCubin);
	dprintf("magic: %x\n", binary->magic);
	dprintf("version: %x\n", binary->version);
	dprintf("data: %p\n", binary->data);
	dprintf("filename_or_fatbins: %p\n", binary->filename_or_fatbins);


	// 调用原函数
	void** (*f)(void *);
	f = dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");

	void** t = f(fatCubin);
	printf("**fatCubinHandle: 0x%llx, *fatCubinHandle: 0x%llx\n", (long long unsigned)t, (long long unsigned)*fatCubinHandle);
	return t;
	// return fatCubinHandle;
}

void sayHi() {
	printf("hi \n");
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

    dprintf("=== __cudaRegisterFunction ===\n");
    dprintf("fatCubinHandle: %p, value: %p\n", fatCubinHandle, *fatCubinHandle);
    dprintf("hostFun: %s (%p)\n", hostFun, hostFun);
    dprintf("deviceFun: %s (%p)\n", deviceFun, deviceFun);
    dprintf("deviceName: %s\n", deviceName);
    dprintf("thread_limit: %d\n", thread_limit);


	computeFatBinaryFormat_t fatBinHeader;
	// fatBinHeader = (computeFatBinaryFormat_t) (*fatCubinHandle);
	
	// 原函数的解法:
	fatBinHeader = (computeFatBinaryFormat_t) (((__fatBinC_Wrapper_t *)(*fatCubinHandle))->data);
	dprintf("headerSize: %lld\n", (long long unsigned)fatBinHeader->headerSize);
	dprintf("fatSize: %lld\n", (long long unsigned)fatBinHeader->fatSize);
	// 头10个
	printf("==head===\n");
	uint8_t* ptr1 = (uint8_t*)fatBinHeader;
	// for(int i=0;i<10;i++) {
	// 	printf("head [0:10] %d: 0x%x \n", i, ptr1[i]);
	// }
	// // 尾10个
	// printf("==tail===\n");
	// uint8_t* ptr2 = &((uint8_t*)fatBinHeader)[fatBinHeader->fatSize - 11];
	// for(int i=0;i<10;i++) {
	// 	printf("tail [-10:] %d: 0x%x \n", i, ptr2[i]);
	// }

	for(int i=0;i<fatBinHeader->fatSize;i++) {
		printf("%d: 0x%x \n", i, ptr1[i]);
	}

	// // dump code
	// FILE *fp = NULL;
	// fp = fopen("./dump.txt", "w+");
	// uint8_t* ptr = (uint8_t*)fatBinHeader;
	// for(int i=0;i<fatBinHeader->fatSize;i++) {
	// 	printf("%d: 0x%x \n", i, ptr[i]);
	// 	fputc(ptr[i], fp);
	// }
	// fclose(fp);

	
	void* ptr = malloc(fatBinHeader->fatSize + fatBinHeader->headerSize);
	// memcpy(ptr, fatBinHeader, fatBinHeader->fatSize);
	memcpy(ptr, fatBinHeader, fatBinHeader->fatSize + fatBinHeader->headerSize);

	cudaRegisterFatbin();
	// loadKernelFunction(fatBinHeader);
	loadKernelFunction(ptr);

  int N = 32;
  int nbytes = N * sizeof(int);
  // int i = 0;
  int *dx = NULL, *hx = NULL;
  // 申请显存
  // cudaMalloc((void**)&dx, nbytes);
  cudaErrorCheck(cuMemAlloc((CUdeviceptr *)&dx, nbytes));

  // 申请成功
  if (dx == NULL) {
    printf("GPU alloc fail");
	return;
  }

  // 申请CPU内存
  hx = (int *)malloc(nbytes);
  if (hx == NULL) {
    printf("CPU alloc fail");
	return;
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
  cudaErrorCheck(cuMemcpyHtoD((CUdeviceptr)dx, hx, nbytes));

  // call GPU
  // sum<<<1, N>>>(dx, dx ,dx);
  void **param = (void **)malloc(sizeof(void *) * 3);
  unsigned int sharedMemBytes = 0;
  CUstream hStream = 0;
  param[0] = &dx;
  param[1] = &dx;
  param[2] = &dx;
  cudaErrorCheck(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, sharedMemBytes, hStream, param, NULL));

  // wait gpu to finish
  cudaThreadSynchronize();

  // copy data to host
  // cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
  cudaErrorCheck(cuMemcpyDtoH(hx, (CUdeviceptr)dx, nbytes));

  printf("hx after:\n");
  for (int i = 0; i < N; i++) {
    printf("%d ", hx[i]);
  }

  printf("\n");
  cudaFree(dx);
  free(hx);

}
 
