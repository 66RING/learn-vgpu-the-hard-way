#include <stdlib.h>
#include <string.h>

#include "faas.h"

/*
 * API自动根据对接的函数所需的参数解析opaque
 * 参数全都保存到cli中了
 */
 
// cudaError_t cudaConfigureCall(
//         dim3 gridDim,
//         dim3 blockDim,
//         size_t sharedMem,
//         cudaStream_t stream);
byte_t handleCudaConfigureCall(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	// 解析参数
	// 处理请求
	// 准备返回

	return reply;
}

// cudaError_t cudaFree (void* devPtr)
byte_t handleCudaFree(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	void **devPtr;

	// 解析参数
	iter = listGetIterator(cli->args, AL_START_HEAD);
	devPtr = (void *)(listNext(iter)->value);

	// 处理请求
	dprintf("server: memfree dst: %p\n", *devPtr);
	// TODO: 暂时用free模拟
	free(*devPtr);

	// 准备返回
	cudaError_t err = 1;
	replyLen = sizeof(cudaError_t);
	replyPtr = reply = newBytes(replyLen);
	memcpy(replyPtr, &err, sizeof(cudaError_t));

	return reply;
}

// char __cudaInitModule(void **fatCubinHandle);
byte_t handle__cudaInitModule(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	// 解析参数
	// 处理请求
	// 准备返回

	return reply;
}

byte_t handleCudaLaunch(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	// 解析参数
	// 处理请求
	// 准备返回

	return reply;
}

// cudaError_t cudaMalloc ( void** devPtr, size_t size )
// 实际参数: size_t size
// 实际返回: cudaError_t err, void* ptr
// TODO: 暂时用malloc模拟
byte_t handleCudaMalloc(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;

	// 解析参数
	size_t *size;
	void* newPtr;
	iter = listGetIterator(cli->args, AL_START_HEAD);
	listNext(iter); // skip devPtr
	size = (size_t *)(listNext(iter)->value);

	// 处理响应
	cudaError_t err = 1;
	newPtr = malloc(sizeof(uint8_t) * *size);
	dprintf("server: malloc size: %lu, return ptr: %p\n", *size, newPtr);

	// 准备返回
	// 格式: cudaError_t, void*
	replyPtr = reply = newBytes(sizeof(cudaError_t) + sizeof(void*));
	memcpy(replyPtr, &err, sizeof(cudaError_t));
	replyPtr += sizeof(cudaError_t);
	memcpy(replyPtr, &newPtr, sizeof(void*));
	replyPtr += sizeof(void*);
	return reply;
}

// cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
// 实际参数: void* dst, void* src, size_t count, cudaMemcpyKind kind
// 返回参数: cudaError_t err
// TODO: 暂时用memcpy模拟
byte_t handleCudaMemcpy(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	void** dst;
	void** src;
	size_t* count;
	cudaMemcpyKind* kind;

	// 解析参数
	iter = listGetIterator(cli->args, AL_START_HEAD);
	dst = (void*)(listNext(iter)->value);
	src = (void*)(listNext(iter)->value);
	count = (size_t*)(listNext(iter)->value);
	kind = (cudaMemcpyKind*)(listNext(iter)->value);

	// 处理请求
	// TODO: err handling
	dprintf("server: memcpy dst: %p, dsc: %p, count: %lu, kind: %d\n", *dst, *src, *count, *kind);
	memcpy(*dst, src, *count);
	dprintf("server get: 0x%x\n", *(int*)*dst);

	// 准备返回
	cudaError_t err = 1;
	replyLen = sizeof(cudaError_t);
	replyPtr = reply = newBytes(replyLen);
	memcpy(replyPtr, &err, sizeof(cudaError_t));

	return reply;
}

// void** __cudaRegisterFatBinary(void *fatCubin);
byte_t handle__cudaRegisterFatBinary(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	// 解析参数
	// 处理请求
	// 准备返回

	return reply;
}

// void __cudaRegisterFunction(
//         void   **fatCubinHandle,
//   const char    *hostFun,
//         char    *deviceFun,
//   const char    *deviceName,
//         int      thread_limit,
//         uint3   *tid,
//         uint3   *bid,
//         dim3    *bDim,
//         dim3    *gDim,
//         int     *wSize
// );
byte_t handle__cudaRegisterFunction(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	// 解析参数
	// 处理请求
	// 准备返回

	return reply;
}

// cudaError_t cudaSetupArgument (const void *arg, size_t size, size_t offset);
byte_t handleCudaSetupArgument(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	// 解析参数
	// 处理请求
	// 准备返回

	return reply;
}


// cudaError_t cudaThreadSynchronize (void)
byte_t handleCudaThreadSynchronize(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	// 解析参数
	// 处理请求
	// 准备返回

	return reply;
}

// void __cudaUnregisterFatBinary(void **fatCubinHandle);
byte_t handle__cudaUnregisterFatBinary(client_t *cli) {
	listIter *iter;
	listNode *node;
	byte_t reply;
	byte_t replyPtr;
	int replyLen;

	// 实际参数
	// 解析参数
	// 处理请求
	// 准备返回

	return reply;
}
