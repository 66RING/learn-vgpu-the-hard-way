#include "faas.h"

// TODO: rebuild
#include "gpu.c"

// 全局server对象
server_t server;

uint8_t* newRequestBuf(enum VGPU_COMMAND cmd, int totalLen) {
	uint8_t *ptr = (uint8_t*)malloc(sizeof(uint8_t) * totalLen);
	*(int*)ptr = cmd;
	return ptr;
}

uint8_t* fillBuffer(void* dst, void* src, int len) {
  *(int*)dst = len;
  dst += sizeof(int);
  memcpy(dst, src, len);
  dst += len;
  return dst;
}

byte_t newBytes(int len) {
  byteHdr *hdr = (byteHdr *)malloc(sizeof(byteHdr) + sizeof(uint8_t) * len);
  hdr->len = len;
  return hdr->data;
}

void freeBytes(byte_t b) {
  if (b == NULL)
    return;
  byteHdr *hdr = (byteHdr *)(b - sizeof(byteHdr));
  free(hdr);
}

// 修改数据的实际大小, 方便接收端处理
void byteResize(byte_t b, int l) {
  byteHdr *hdr = (byteHdr *)(b - sizeof(byteHdr));
  hdr->len = l;
}

// 返回buffer长度, 不一定是数据实际长度
// 长度不包含元数据长度
int byteLen(byte_t b) {
  if (b == NULL)
    return -1;
  byteHdr *hdr = (byteHdr *)(b - sizeof(byteHdr));
  return hdr->len;
}

void byteInspect(byte_t b) {
  byteHdr *hdr = (byteHdr *)(b - sizeof(byteHdr));
  for (int i = 0; i < hdr->len; i++) {
    printf("0x%x ", hdr->data[i]);
  }
}

void *byteStream(byte_t b) {
  if (b == NULL)
    return NULL;
  return (void *)(b - sizeof(byteHdr));
}

// 将matedata和数据一并发送到fd
int writeStream(int fd, byte_t b) {
  byteHdr *hdr = (byteHdr *)byteStream(b);
  return write(fd, hdr, sizeof(byteHdr) + hdr->len);
}

// 新建client对象
client_t *createClient() {
  client_t *cli = (client_t *)malloc(sizeof(client_t));
  cli->requestBuf = (uint8_t *)malloc(sizeof(uint8_t) * BUFFER_SIZE);
  cli->reply = listCreate();
  cli->fd = 0;
  cli->requestLen = 0;
  cli->indexPtr = 0;
  cli->cmdType = COMMAND_UNKNOWN;
  cli->args = listCreate();
  cli->argc = 0;
  return cli;
}

void freeClient(client_t *cli) {
  listRelease(cli->args);
  listRelease(cli->reply);
  free(cli->requestBuf);
  free(cli);
}

void freeArgs(list *args) {}

void resetClient(client_t *cli) {
  cli->indexPtr = 0;
  cli->requestLen = 0;
  cli->cmdType = COMMAND_UNKNOWN;
  cli->argc = 0;
  freeArgs(cli->args);
}

// 向客户端发送回复
void sendReplyToClient(struct aeEventLoop *eventLoop, int fd, void *clientData,
                       int mask) {
  client_t *cli = (client_t *)clientData;
  dprintf("sendReplyToClient, reply len: %d\n", listLength(cli->reply));

  listIter *iter;
  listNode *node;
  list *list = cli->reply;
  // 尽可能多的发送积攒的回复
  iter = listGetIterator(list, AL_START_HEAD);
  while ((node = listNext(iter)) != NULL) {
    byte_t b = (byte_t)node->value;
    // TODO: 假设能够一次发完因为目前是单线程, 单客户端
    int n = writeStream(fd, b);
    if (n != byteLen(b) + sizeof(byteHdr)) {
      printf("FAIL: sendReplyToClient fail len");
      close(fd);
      exit(1);
    }
	// 回复完毕, 删除reply节点
	listDelNode(cli->reply, node);
  }

  // 释放迭代器所占用的内存空间
  listReleaseIterator(iter); 
  aeDeleteFileEvent(server.loop, fd, AE_WRITABLE);
  // client状态重置
  // 现在重置是因为对于一个client肯定就是一个连接, 不会产生并发情况, 所以发送后再重置
  resetClient(cli);
}

// 向回复列表中插入新回复
// 注册发送事件, 带写epoll就绪就发送数据
void addReply(client_t *cli, byte_t reply) {
  listAddNodeTail(cli->reply, reply);
  aeCreateFileEvent(server.loop, cli->fd, AE_WRITABLE, sendReplyToClient, cli,
                    NULL);
}

void proccessCommand(client_t *cli) {
  // TODO: 实际处理。这里先打印参数信息
  listIter *iter;
  listNode *node;
  list *list = cli->args;

  DEBUG_BLOCK(
	int i = 0; iter = listGetIterator(list, AL_START_HEAD);
	while ((node = listNext(iter)) != NULL) {
	  printf("arg%d: ", i);
      byteInspect((byte_t)node->value);
      printf("\n");
      i++;
	}
  );

  byte_t reply;
  switch (cli->cmdType) {
  case VGPU_CUDA_MALLOC:
	reply = handleCudaMalloc(cli);
    break;
  case VGPU_CUDA_FREE:
	reply = handleCudaFree(cli);
    break;
  case VGPU_CUDA_MEMCPY:
	reply = handleCudaMemcpy(cli);
    break;
  case VGPU_CUDA_REGISTER_FAT_BINARY:
    break;
  case VGPU_CUDA_REGISTER_FUNCTION:
    break;
  case VGPU_CUDA_KERNEL_LAUNCH:
    break;
  case VGPU_CUDA_THREAD_SYNCHRONIZE:
    break;
  default:
    printf("cmd unknow");
	exit(1);
  }

  addReply(cli, reply);
}

// 命令完整返回true
// 处理buf, 将参数保存到client对象中
int handleBuf(client_t *cli) {
  while (cli->argc > 0) {
    // 处理一段buf, 消耗一段buf
    // buf格式: (dataLen, data), (dataLen, data)...

    // TODO: 假设数据一次性读取完成
    byteHdr *bhdr;
    bhdr = (byteHdr *)&cli->requestBuf[cli->indexPtr];

    byte_t dataStore = newBytes(bhdr->len);
    cli->indexPtr += sizeof(int);
    memcpy(dataStore, &cli->requestBuf[cli->indexPtr], bhdr->len);

    // update buf
    cli->indexPtr += bhdr->len;
    listAddNodeTail(cli->args, dataStore);
    cli->argc--;
  }
  return 0;
}

// TODO: 返回命令参数的个数
int countArgc(int cmdType) {
  switch (cmdType) {
  case VGPU_CUDA_MALLOC:
    return 2;
    break;
  case VGPU_CUDA_FREE:
	return 1;
    break;
  case VGPU_CUDA_MEMCPY:
	return 4;
    break;
  case VGPU_CUDA_REGISTER_FAT_BINARY:
	return 1;
    break;
  case VGPU_CUDA_REGISTER_FUNCTION:
	return 10;
    break;
  case VGPU_CUDA_KERNEL_LAUNCH:
	// TODO
	exit(1);
    break;
  case VGPU_CUDA_THREAD_SYNCHRONIZE:
	return 0;
    break;
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
    proccessCommand(cli);
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

  aeMain(loop);
}

void startServerThread() {
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
