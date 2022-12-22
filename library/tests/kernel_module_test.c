#include<sys/ioctl.h>
#include<stdlib.h>
#include<fcntl.h>
#include<inttypes.h>
#include<stdlib.h>
#include<string.h>
#include<sys/syscall.h>

#include "../../protocol/vgpu_common.h"

const char dev_path[] = "/dev/vgpu";
int fd = -1;

static void device_open() {
  fd = open(dev_path, O_RDWR);
  if (fd < 0) {
    exit(EXIT_FAILURE);
  }
}

static void send_command(VgpuArgs *args) {
  if (fd < 0)
    device_open();
  ioctl(fd, args->cmd, args);
}


void dummy_send() {
	VgpuArgs args;
	memset(&args, 0, sizeof(VgpuArgs));
	// TODO:
	send_command(&args);
}


// 申请内存资源, 并赋值到devPtr
void cudaMalloc(void **devPtr, size_t size) {
	VgpuArgs args;
	memset(&args, 0, sizeof(VgpuArgs));

	args.cmd = VGPU_CUDA_MALLOC;
	args.owner_id = syscall(__NR_gettid);
	// TODO:
	send_command(&args);
	*devPtr = (void*)args.dst;
	printf("0x%x\n", (int)args.dst);
}


void ioctl_test() {
	dummy_send();
}

void echo_test() {
	int *ptr = malloc(sizeof(int));
	cudaMalloc(&ptr, 0);
}

int main() {
	ioctl_test();
	echo_test();
}
