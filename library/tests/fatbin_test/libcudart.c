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

// cuda9: /usr/local/cuda/include/fatbinary.h
// typedef struct fatBinaryHeader *computeFatBinaryFormat_t
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

    dprintf("=== __cudaRegisterFunction ===\n");
    dprintf("fatCubinHandle: %p, value: %p\n", fatCubinHandle, *fatCubinHandle);
    dprintf("hostFun: %s (%p)\n", hostFun, hostFun);
    dprintf("deviceFun: %s (%p)\n", deviceFun, deviceFun);
    dprintf("deviceName: %s\n", deviceName);
    dprintf("thread_limit: %d\n", thread_limit);


	computeFatBinaryFormat_t fatBinHeader;
	fatBinHeader = (computeFatBinaryFormat_t) (*fatCubinHandle);

	/// debug
	/// 和后端收到的fatBin比对前80字节
	dprintf("fatBin dump code (size = %lld): \n", fatBinHeader->fatSize);
	uint32_t* ptr = (uint32_t*)fatBinHeader;
	for(int i=0;i<10;i++) {
		printf("0x%x ", ptr[i]);
	}
	printf("\n");
	///
}
 
