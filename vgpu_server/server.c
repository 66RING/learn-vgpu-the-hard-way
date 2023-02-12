#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "faas.h"

const int BACKLOG = 64;

int tcpServer(int port) {
  int sockfd;
  int on = 1;
  struct sockaddr_in servaddr, cli;

  // 创建socket
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    printf("socket creation failed...\n");
    return -1;
  }

  // 启用复用
  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
    printf("socket opt failed...\n");
    close(sockfd);
    return -1;
  }

  // 设置IP和端口
  bzero(&servaddr, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  servaddr.sin_port = htons(port);

  // 绑定socket和服务描述符
  if ((bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr))) != 0) {
    printf("socket bind failed...\n");
    return -1;
  }

  // 启动监听
  if ((listen(sockfd, BACKLOG)) != 0) {
    printf("Listen failed...\n");
    return -1;
  }

  return sockfd;
}
