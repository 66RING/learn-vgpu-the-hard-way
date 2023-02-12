// #include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <thread>

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

#include <list>
#include <vector>

#include "ae.h"
#include "faas.h"
#include "anet.h"

#define PORT 8888
#define BUFFER_SIZE 4*1024*1024
#define COMMAND_UNKNOWN -1

using namespace std;

#if 0
	#define dprintf(fmt, arg...) printf("DEBUG: "fmt, ##arg)
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

// typedef struct {
//   uint8_t* buf;
//   int len;
// } Byte;

class Byte {
public:
  Byte(int n) {
	buf_ = (uint8_t*)malloc(sizeof(uint8_t) * n);
	len_ = n;
  }
  ~Byte() {
	free(buf_);
  }
  void show() {
	for (int i = 0; i < len_; i++) {
	  printf("%d ", buf_[i]);
	}
  }

  uint8_t* at(int n) {
	return &buf_[n];
  }
private:
  int len_;
  uint8_t* buf_;
};

class Client {
public:
  Client() {
	buf_ = (uint8_t*)malloc(sizeof(uint8_t) * BUFFER_SIZE);

  };
  ~Client() {
	free(buf_);
  };
  int fd_;
  struct sockaddr_in client;
  // buf末尾指针
  int requestLen {0};
  // 已经处理的长度
  int indexPtr {0};
  // 当前流式处理的命令类型
  int cmdType;
  vector<Byte*> args;
  int argc;

  // TODO: 目前缓冲区固定长度
  // Byte buf_;
  uint8_t* buf_;

private:

};

class VgpuServer {
public:
  VgpuServer() = default;
  VgpuServer(int fd, int port);

  void SetLoop(aeEventLoop* loop) {
	loop_ = loop;
  }

  void SetFd(int fd) {
	fd_ = fd;
  }

  aeEventLoop* GetLoop() {
	return loop_;
  }

  int GetFd() {
	return fd_;
  }

  aeEventLoop *loop_;
private:
  int fd_;
  int port_;
  list<Client*> clients_;
};

VgpuServer server;

VgpuServer::VgpuServer(int fd, int port): fd_(fd), port_(port) {
  loop_ = aeCreateEventLoop();
}

int countArgc(int cmdType) {
  // TODO:计算命令有几个参数
  return -1;
}

void processCommand(Client* cli) {
  // TODO: 实际处理。这里先打印参数信息
  for (int i = 0; i < cli->args.size(); i++) {
	printf("arg %d: ", i);
	cli->args[i]->show();
	printf("\n");
  }
}

// 命令完整返回true
// 处理buf, 将参数保存到client对象中
bool handleBuf(Client* cli) {
  while (cli->argc > 0) {
	// 处理一段buf, 消耗一段buf
	// buf格式: (dataLen, data), (dataLen, data)...

	// TODO: 假设数据一次性读取完成
	ItemHdr *ihdr;
	ihdr = (ItemHdr*)&cli->buf_[cli->indexPtr];

	Byte* data = new Byte(ihdr->len);
	memcpy(data->at(0), &cli->buf_[cli->indexPtr], ihdr->len);

	// update buf
	cli->indexPtr += ihdr->len;
	cli->args.push_back(data);
	cli->argc--;
  }
  return true;
}

// 处理并执行
// 如果命令不完整将待读取完毕再执行
// 传输协议: $ + cmdType\r\n + data1\r\n + data2\r\n ...
void proccessBuf(Client* cli) {
  if (cli->cmdType == COMMAND_UNKNOWN) {
	// 第一次处理
	RPCHdr *h = (RPCHdr*)cli->buf_;
	cli->cmdType = h->cmdType;
	cli->argc = countArgc(cli->cmdType);
	cli->args.clear();
	cli->indexPtr += sizeof(RPCHdr);
  }

  int ok = handleBuf(cli);
  if (ok) {
	// 命令完整可以处理
	processCommand(cli);
  }
}

// 命令分发的入口
// 协议内容: '$' + 命令id + '\r\n' + 若干参数\r\n
// @param clientData: client, 接收传入的client对象
// @param fd: clientfd, 连接的fd
// TODO: 流式处理, 因为可能一下子读不完
void requestFromClient(struct aeEventLoop *eventLoop, int fd, void *clientData, int mask) {
  Client* cli = (Client*)clientData;
  int n;
  // 读取
  n = read(fd, cli->buf_ + cli->requestLen, BUFFER_SIZE);
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
  cli->requestLen += n;
  dprintf("request from client %d bytes\n", n);
  proccessBuf(cli);
}

void acceptHandler(struct aeEventLoop *eventLoop, int fd, void *clientData, int mask) {
  Client* cli = new Client();
  int len = sizeof(cli->client);
  int clientfd = ::accept(fd, (struct sockaddr*)&cli->client, (socklen_t*)&len);
  if (clientfd == -1) {
	printf("accept error");
	close(fd);
	exit(1);
  }

  // 初始化client
  cli->fd_ = clientfd;

  dprintf("server accepted client's connection\n");
  aeCreateFileEvent(eventLoop, clientfd, AE_READABLE, requestFromClient, cli, nullptr);
}

void _startServer(VgpuServer* server) {
  aeMain(server->loop_);
}

void startServer() {
  // 创建主循环用于服务器的连接监听
  server.SetLoop(aeCreateEventLoop());
  // 创建服务器
  server.SetFd(tcpServer(PORT));
  // 为服务器对象绑定事件循环
  aeCreateFileEvent(server.GetLoop(), server.GetFd(), AE_READABLE, acceptHandler, nullptr, nullptr);

  thread serverThread(_startServer, &server);
}

int main() { return 0; }
