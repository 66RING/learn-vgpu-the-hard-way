#include <stdlib.h>
#include <string.h>

#include "faas.h"


// cudaConfigureCall@@libcudart.so.9.0
// cudaFree@@libcudart.so.9.0
// __cudaInitModule@@libcudart.so.9.0
// cudaLaunch@@libcudart.so.9.0
// cudaMalloc@@libcudart.so.9.0
// cudaMemcpy@@libcudart.so.9.0
// __cudaRegisterFatBinary@@libcudart.so.9.0
// __cudaRegisterFunction@@libcudart.so.9.0
// cudaSetupArgument@@libcudart.so.9.0
// cudaThreadSynchronize@@libcudart.so.9.0
// __cudaUnregisterFatBinary@@libcudart.so.9.0


/*
 * API自动根据对接的函数所需的参数解析opaque
 * 参数全都保存到cli中了
 */

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

byte_t handleCudaFree(client_t *cli) {
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
	dprintf("malloc size: %lu, return ptr: %p\n", *size, newPtr);

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
	cudaError_t err = 1;
	dprintf("memcpy dst: %p, dsc: %p, count: %lu, kind: %d\n", *dst, *src, *count, *kind);
	memcpy(*dst, src, *count);
	printf("0x%x\n", *(int*)*dst);

	// 准备返回
	replyLen = sizeof(cudaError_t);
	replyPtr = reply = newBytes(replyLen);
	memcpy(replyPtr, &err, sizeof(cudaError_t));

	return reply;
}

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
