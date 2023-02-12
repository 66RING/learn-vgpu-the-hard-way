#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "adlist.h"
#include "ae.h"
#include "anet.h"
#include "faas.h"

#define PORT 8888
#define BUFFER_SIZE 4 * 1024 * 1024
#define COMMAND_UNKNOWN -1
#define CMD_TEST 1

#if 0
#define dprintf(fmt, arg...) printf("DEBUG: " fmt, ##arg)
#else
#define dprintf(fmt, arg...)
#endif

typedef struct {
  int cmdType;
  char data[];
} RPCHdr;

typedef struct {
  int len;
  char data[];
} ItemHdr;

typedef struct {
  uint8_t *buf;
  int bufLen;
} byte_t;

byte_t *newBytes(int len) {
  byte_t *b = (byte_t *)malloc(sizeof(byte_t));
  b->bufLen = len;
  b->buf = (uint8_t *)malloc(sizeof(uint8_t) * len);
  return b;
}

void freeBytes(byte_t *b) {
  free(b->buf);
  free(b);
}

uint8_t *byteSlice(byte_t *b, int idx) { return b->buf + idx; }

int byteLen(byte_t *b) { return b->bufLen; }

void byteInspect(byte_t *b) {
  for (int i = 0; i < b->bufLen; i++) {
    printf("0x%x ", b->buf[i]);
  }
}

// TODO:流式read/write, 可能网络一次写/读补完
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
  // 准备就绪的reply
  uint8_t *reply;
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

server_t server;

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

// TODO:
client_t *createClient() {
  client_t *cli = (client_t *)malloc(sizeof(client_t));
  cli->requestBuf = (uint8_t *)malloc(sizeof(uint8_t) * BUFFER_SIZE);
  cli->reply = (uint8_t *)malloc(sizeof(uint8_t) * BUFFER_SIZE);
  cli->fd = 0;
  cli->requestLen = 0;
  cli->indexPtr = 0;
  cli->cmdType = -1;
  cli->args = listCreate();
  cli->argc = 0;
  // TODO: cli init things
  return cli;
}

void freeClient(client_t *cli) {
  free(cli->requestBuf);
  free(cli->reply);
  free(cli);
}

void processCommand(client_t *cli) {
  // TODO: 实际处理。这里先打印参数信息
  listIter *iter;
  listNode *node;
  list *list = cli->args;

  int i = 0;
  iter = listGetIterator(list, AL_START_HEAD);
  while ((node = listNext(iter)) != NULL) {
	  printf("arg%d: ", i);
    byteInspect((byte_t *)node->value);
	printf("\n");
	i++;
  }
  listReleaseIterator(iter); // 释放迭代器所占用的内存空间
}

// 命令完整返回true
// 处理buf, 将参数保存到client对象中
int handleBuf(client_t *cli) {
  while (cli->argc > 0) {
    // 处理一段buf, 消耗一段buf
    // buf格式: (dataLen, data), (dataLen, data)...

    // TODO: 假设数据一次性读取完成
    ItemHdr *ihdr;
    ihdr = (ItemHdr *)&cli->requestBuf[cli->indexPtr];

    byte_t *data = newBytes(ihdr->len);
    cli->indexPtr += sizeof(int);
    memcpy(byteSlice(data, 0), &cli->requestBuf[cli->indexPtr], ihdr->len);

    // update buf
    cli->indexPtr += ihdr->len;
    listAddNodeTail(cli->args, data);
    cli->argc--;
  }
  return 0;
}

int countArgc(int cmdType) {
  switch (cmdType) {
  case CMD_TEST:
    return 2;
  default:
    printf("cmdType undefine");
    exit(1);
  }
  // TODO:计算命令有几个参数
  return -1;
}

// 流式处理, 检查buf是否完成, 然后进行下一步处理
// TODO:
void proccessBuf(client_t *cli) {
  if (cli->cmdType == COMMAND_UNKNOWN) {
    // 第一次处理
    RPCHdr *h = (RPCHdr *)cli->requestBuf;
    cli->cmdType = h->cmdType;
    cli->argc = countArgc(cli->cmdType);
    listRelease(cli->args);
    cli->args = listCreate();
    cli->indexPtr += sizeof(RPCHdr);
  }

  int ok = handleBuf(cli);
  if (ok != -1) {
    // 命令完整可以处理
    processCommand(cli);
  }
}

// 命令分发的入口
// 协议内容: '$' + 命令id + '\r\n' + 若干参数\r\n
// @param clientData: client, 接收传入的client对象
// @param fd: clientfd, 连接的fd
// TODO: 流式处理, 因为可能一下子读不完
void requestFromClient(struct aeEventLoop *eventLoop, int fd, void *clientData,
                       int mask) {
  client_t *cli = (client_t *)clientData;
  int n;

  // 假设一次可以读取完毕
  n = read(fd, cli->requestBuf, BUFFER_SIZE);
  if (n == BUFFER_SIZE) {
    dprintf("FAIL: buffer overflow\n");
    close(fd);
    exit(1);
  }
  if (n == -1) {
    dprintf("FAIL: read requestFromClient\n");
    close(fd);
    exit(1);
  }
  if (n < sizeof(int)) {
    dprintf("FAIL: read header error\n");
    close(fd);
    exit(1);
  }

  // TODO: 假设一次能读完, 所以直接 = n
  cli->requestLen = n;
  dprintf("request from client %d bytes\n", n);
  proccessBuf(cli);
}

// 接收client的连接
void acceptHandler(struct aeEventLoop *eventLoop, int fd, void *clientData,
                   int mask) {
  client_t *cli = createClient();
  int len = sizeof(cli->cli);
  int cfd = accept(fd, (struct sockaddr *)&cli->cli, (socklen_t *)&len);
  if (cfd == -1) {
    printf("accept error");
    close(fd);
    exit(1);
  }

  // 初始化连接的client
  cli->fd = cfd;

  dprintf("server accepted client's connection\n");
  aeCreateFileEvent(eventLoop, cfd, AE_READABLE, requestFromClient, cli, NULL);
}

void _startServer(void *arg) {
  aeEventLoop *loop = arg;
  aeMain(loop);
}

void startServer() {
  aeEventLoop *loop;
  int server_fd;

  // 创建主循环用于服务器的连接监听
  loop = aeCreateEventLoop();
  // 创建服务器
  server_fd = tcpServer(PORT);
  // 为服务器对象绑定事件循环
  aeCreateFileEvent(loop, server_fd, AE_READABLE, acceptHandler, NULL, NULL);

  // 初始化server对象
  server.clients = listCreate();
  server.loop = loop;
  server.fd = server_fd;

  // 启动server测试
  pthread_t tid;
  pthread_create(&tid, NULL, (void *)_startServer, loop);
}

int main() {
  startServer();

  // test

  // 传输协议: cmdType: int, (dataLen: int, data)...
  // 启动client
  char err[256];
  char buf[256];
  int client_fd = anetTcpConnect(err, "0.0.0.0", PORT);

  char arg1[] = "hi\0";
  char arg2[] = "world\0";
  uint8_t *msg;
  uint8_t *ptr;
  int len1 = strlen(arg1);
  int len2 = strlen(arg2);
  int totalLen = sizeof(int) + sizeof(size_t) + sizeof(char) * strlen(arg1) +
                 sizeof(size_t) + sizeof(char) * strlen(arg2);
  msg = (uint8_t *)malloc(totalLen);
  ptr = msg;
  *(int *)ptr = CMD_TEST;
  ptr += sizeof(int);

  *(int *)ptr = len1;
  ptr += sizeof(int);
  memcpy(ptr, &arg1, len1);
  ptr += len1;

  *(int *)ptr = len2;
  ptr += sizeof(int);
  memcpy(ptr, &arg2, len2);
  ptr += len2;

  int n = write(client_fd, msg, totalLen);
  dprintf("client sent done\n");
  if (n != totalLen) {
    printf("FAIL: write len error: %d\n", n);
    exit(1);
  }

  // n = read(client_fd, buf, 256);
  // dprintf("client read done\n");
  // if (n != 10) {
  // printf("FAIL: read len error: %d\n", n);
  // exit(1);
  // }

  getchar();
  server.loop->stop = 1;
  dprintf("server stop\n");
  // printf("PASS");

  return 0;
}
