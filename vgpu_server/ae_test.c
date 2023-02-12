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

#define PORT 8888

#if 0
	#define dprintf(fmt, arg...) printf("DEBUG: "fmt, ##arg)
#else
	#define dprintf(fmt, arg...)
#endif


void writeProc(struct aeEventLoop *eventLoop, int fd, void *clientData, int mask) {
	char *buf = clientData;
	dprintf("server write: %s\n", buf);
	int n = write(fd, buf, strlen(buf));
	if (n != 10) {
		printf("FAIL: server write len error %d\n", n);
		close(fd);
		exit(1);
	}

	aeDeleteFileEvent(eventLoop, fd, AE_WRITABLE);
	free(buf);
}

void readProc(struct aeEventLoop *eventLoop, int fd, void *clientData, int mask) {
	char *buf = (char*)malloc(sizeof(char) * 20);
	int n = read(fd, buf, 20);
	buf[n] = '\0';
	dprintf("server read: %s\n", buf);
	if (n != 10) {
		printf("FAIL: server read len error %d\n", n);
		close(fd);
		exit(1);
	}
	aeCreateFileEvent(eventLoop, fd, AE_WRITABLE, writeProc, buf, NULL);
}

//: echo server here
void acceptProc(struct aeEventLoop *eventLoop, int fd, void *clientData, int mask) {
	struct sockaddr_in cli;
	int len = sizeof(cli);
	int cfd = accept(fd, (struct sockaddr*)&cli, &len);
	if (cfd == -1) {
		printf("accept error");
		close(fd);
		exit(1);
	}
	dprintf("server accepted client's connection\n");
	aeCreateFileEvent(eventLoop, cfd, AE_READABLE, readProc, clientData, NULL);
}


void startServer(void *arg) {
	aeEventLoop *loop = arg;
	aeMain(loop);
}

int main() {
	aeEventLoop *loop;
	int server_fd;

	// 创建主循环用于服务器的连接监听
	loop = aeCreateEventLoop();
	// 创建服务器
	server_fd = tcpServer(PORT);
	// 为服务器对象绑定事件循环
	aeCreateFileEvent(loop, server_fd, AE_READABLE, acceptProc, NULL, NULL);

	// 启动server测试
	pthread_t tid;
	pthread_create(&tid, NULL, (void *)startServer, loop);


	// 启动client
	char err[256];
	char msg[] = "helloworld\0";
	char buf[256];
	int client_fd = anetTcpConnect(err, "0.0.0.0", PORT);

	int n = write(client_fd, msg, 10);
	dprintf("client sent done\n");
	if (n != 10) {
		printf("FAIL: write len error: %d\n", n);
		exit(1);
	}

	n = read(client_fd, buf, 256);
	dprintf("client read done\n");
	if (n != 10) {
		printf("FAIL: read len error: %d\n", n);
		exit(1);
	}


	loop->stop = 1;
	dprintf("server stop\n");
	printf("PASS");
}
