#include "faas.h"

uint8_t* newRequestBuf(enum VGPU_COMMAND cmd, int totalLen);
uint8_t* fillBuffer(void* dst, void* src, int len);

byte_t newBytes(int len);
void freeBytes(byte_t b);

// 修改数据的实际大小, 方便接收端处理
void byteResize(byte_t b, int l); 

// 返回buffer长度, 不一定是数据实际长度
// 长度不包含元数据长度
int byteLen(byte_t b); 

void byteInspect(byte_t b); 

void *byteStream(byte_t b); 

// 将matedata和数据一并发送到fd
int writeStream(int fd, byte_t b); 

// // 新建client对象
// client_t *createClient(); 

// void freeClient(client_t *cli); 

// void freeArgs(list *args); 

// void resetClient(client_t *cli); 

// // 向客户端发送回复
// void sendReplyToClient(struct aeEventLoop *eventLoop, int fd, void *clientData,
//                        int mask); 

// // 向回复列表中插入新回复
// // 注册发送事件, 带写epoll就绪就发送数据
// void addReply(client_t *cli, byte_t reply); 

// void proccessCommand(client_t *cli); 

// // 命令完整返回true
// // 处理buf, 将参数保存到client对象中
// int handleBuf(client_t *cli); 

// // TODO: 返回命令参数的个数
// int countArgc(int cmdType); 

// // 流式处理, 检查buf是否完成, 然后进行下一步处理
// // TODO:
// void proccessBuf(client_t *cli); 

// // 命令分发的入口
// // 协议内容: '$' + 命令id + '\r\n' + 若干参数\r\n
// // @param clientData: client, 接收传入的client对象
// // @param fd: clientfd, 连接的fd
// // TODO: 流式处理, 因为可能一下子读不完
// void requestFromClient(struct aeEventLoop *eventLoop, int fd, void *clientData,
//                        int mask); 

// // 接收client的连接
// void acceptHandler(struct aeEventLoop *eventLoop, int fd, void *clientData,
//                    int mask); 

// 启动主循环
void startServer(); 

// 启动server线程
void startServerThread(); 
