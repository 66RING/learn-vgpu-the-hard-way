#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <stdlib.h>

// from cuda header
#include <builtin_types.h>


#include "../protocol/vgpu_common.h"

#define error(fmt, arg...) printf("ERROR: "fmt, ##arg)

const char dev_path[] = "/dev/qcuda";
int fd = -1;

static void device_open() {
  fd = open(dev_path, O_RDWR);
  if (fd < 0) {
    error("open device %s faild, %s (%d)\n", dev_path, strerror(errno), errno);
    exit(EXIT_FAILURE);
  }
}

static void send_command(enum VGPU_COMMAND cmd, VgpuArgs *args) {
  if (fd < 0)
    device_open();
  ioctl(fd, cmd, args);
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
	// TODO:
	send_command(VGPU_CUDA_MALLOC, NULL);
	return (cudaError_t)0;
}
