#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>

#include "ae.h"
#include "faas.h"
#include "anet.h"
#include "vgpu_server.h"

extern server_t server;

void testCudaMem() {
  startServerThread();

  // test

  // **传输协议: cmdType: int, (dataLen: int, data: uint8_t[])...**
  // 启动client
  char err[256];
  char buf[256];
  int client_fd = anetTcpConnect(err, "0.0.0.0", PORT);

  void* devPtr;
  {
	// args
	size_t size = 10;
	int len1 = sizeof(void*);
	int len2 = sizeof(size_t);

	uint8_t *msg;
	uint8_t *ptr;
	int totalLen = sizeof(enum VGPU_COMMAND)
		+ sizeof(int) + len1
		+ sizeof(int) + len2;

	msg = ptr = newRequestBuf(VGPU_CUDA_MALLOC, totalLen);
	ptr += sizeof(enum VGPU_COMMAND);

	ptr = fillBuffer(ptr, &devPtr, len1);
	ptr = fillBuffer(ptr, &size, len2);

	int n = write(client_fd, msg, totalLen);
	dprintf("client sent done\n");
	if (n != totalLen) {
	  printf("FAIL: write len error: %d\n", n);
	  exit(1);
	}

	byteHdr *bhdr = byteStream(newBytes(256));
	byte_t b = bhdr->data;
	n = read(client_fd, bhdr, 256);
	byteResize(b, bhdr->len);
	dprintf("sizeof void* %lu\n", sizeof(void*));
	dprintf("client read done, len: %d, expect len: %lu\n", bhdr->len, sizeof(void*) + sizeof(cudaError_t));

	// byteInspect(b);
	ptr = b;
	printf("err code: %d\n", *(cudaError_t*)ptr);
	ptr += sizeof(cudaError_t);
	printf("malloc: %p\n", *(void**)ptr);
	devPtr = *(void**)ptr;
	ptr += sizeof(void*);
  }

  {
	void* dst;
	dst = devPtr;
	int len1 = sizeof(void*);

	int src = 0xdeadbeef;
	int len2 = sizeof(int);

	size_t count = sizeof(int);
	int len3 = sizeof(size_t);

	cudaMemcpyKind kind = 1;
	int len4 = sizeof(cudaMemcpyKind);

	uint8_t *msg;
	uint8_t *ptr;
	int totalLen = sizeof(enum VGPU_COMMAND)
	  + sizeof(int) + len1
	  + sizeof(int) + len2
	  + sizeof(int) + len3
	  + sizeof(int) + len4;

	// ptr = msg = (uint8_t*)malloc(sizeof(uint8_t) * totalLen);
	// *(int*)ptr = VGPU_CUDA_MEMCPY;
	// ptr += sizeof(int);
	msg = ptr = newRequestBuf(VGPU_CUDA_MEMCPY, totalLen);
	ptr += sizeof(enum VGPU_COMMAND);

	// 像src这种变长数据应该怎样传递给后端
	// 这就是为什么发送的协议要是(len, data)...
	// 当然返回的协议可能也是要的: memcpy拷贝回
	ptr = fillBuffer(ptr, &dst, len1);
	ptr = fillBuffer(ptr, &src, len2);
	ptr = fillBuffer(ptr, &count, len3);
	ptr = fillBuffer(ptr, &kind, len4);
	printf("memcpy src: 0x%x, dst: %p, count: %lu, kind: %d\n", src, dst, count, kind);

	int n = write(client_fd, msg, totalLen);
	dprintf("client sent done\n");
	if (n != totalLen) {
	  printf("FAIL: write len error: %d\n", n);
	  exit(1);
	}

	byteHdr *bhdr = byteStream(newBytes(256));
	byte_t b = bhdr->data;
	n = read(client_fd, bhdr, 256);
	byteResize(b, bhdr->len);
  }

  {
	void* dst;
	dst = devPtr;
	int len1 = sizeof(void*);

	uint8_t *msg;
	uint8_t *ptr;
	int totalLen = sizeof(enum VGPU_COMMAND)
	  + sizeof(int) + len1;

	msg = ptr = newRequestBuf(VGPU_CUDA_FREE, totalLen);
	ptr += sizeof(enum VGPU_COMMAND);

	// 像src这种变长数据应该怎样传递给后端
	// 这就是为什么发送的协议要是(len, data)...
	// 当然返回的协议可能也是要的: memcpy拷贝回
	ptr = fillBuffer(ptr, &dst, len1);
	dprintf("client: memfree dst: %p\n", dst);

	int n = write(client_fd, msg, totalLen);
	dprintf("client sent done\n");
	if (n != totalLen) {
	  printf("FAIL: write len error: %d\n", n);
	  exit(1);
	}

	byteHdr *bhdr = byteStream(newBytes(256));
	byte_t b = bhdr->data;
	n = read(client_fd, bhdr, 256);
	byteResize(b, bhdr->len);
  }

  getchar();
  server.loop->stop = 1;
  dprintf("server stop\n");
  // printf("PASS");
}

int main() {
	testCudaMem();

	return 0;
}

