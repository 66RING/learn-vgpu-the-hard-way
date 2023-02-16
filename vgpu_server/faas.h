#ifndef FAAS_H
#define FAAS_H

#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <string.h>
#include <sys/socket.h>

#include "adlist.h"
#include "ae.h"
#include "anet.h"

#include "../protocol/vgpu_common.h"

// TODO: 暂时模拟一下 cuda header
typedef uint32_t cudaError_t;
typedef int cudaMemcpyKind;

#if 1
#define dprintf(fmt, arg...) printf("DEBUG: " fmt, ##arg)
#else
#define dprintf(fmt, arg...)
#endif

#if 1
	#define DEBUG_BLOCK(x) do { x }  while(0)
#else
	#define DEBUG_BLOCK(x) 
#endif 


#define BUFFER_SIZE 4 * 1024 * 1024

typedef struct {
  int cmdType;
  char data[];
} RPCHdr;

// 用户不直接使用, 用于二进制安全的网络数据传递
typedef struct {
  int len;
  uint8_t data[]; // 不占用空间
} byteHdr;

// 
typedef uint8_t* byte_t;

// TODO:流式read/write, 可能网络一次写/读不完
typedef struct {
  int fd;
  struct sockaddr_in cli;
  // TODO: 固定大小读缓存, 假设一次性能够读完
  uint8_t *requestBuf;
  // buf末尾指针
  int requestLen;
  // 已经处理的长度
  int indexPtr;
  // 当前流式处理的命令类型
  int cmdType;
  // 准备就绪的reply, item: uint8_t* ptr
  list *reply;
  // 准备就绪的args
  list *args;
  int argc;
} client_t;

typedef struct {
  // epoll fd
  int fd;
  list *clients;
  int port;
  aeEventLoop *loop;
} server_t;


















byte_t newBytes(int len);
void freeBytes(byte_t b);
void byteResize(byte_t b, int l);
int byteLen(byte_t b);
void byteInspect(byte_t b);
void* byteStream(byte_t b);
int writeStream(int fd, byte_t b);


// create a tcp for testing
// return fd
int tcpServer(int port);


#endif
