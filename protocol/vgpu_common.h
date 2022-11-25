#ifndef VGPU_COMMON_H
#define VGPU_COMMON_H

#define VIRTIO_ID_VGPU 30

typedef struct VgpuArgs VgpuArgs;

enum VGPU_COMMAND
{
  VGPU_OPEN_COMMAND,
  VGPU_RELEASE_COMMAND,
  VGPU_MMAP_COMMAND,
  VGPU_CUDA_MALLOC,
  CMD_MAX,
};

enum VGPU_DIRECTION
{
  D2H,
  H2D,
  D2D,
  H2H,
};

struct VgpuArgs {
  enum VGPU_COMMAND cmd;
  uint64_t src;
  uint64_t src_size;
  uint64_t dst;
  uint64_t dst_size;

  uint64_t owner_id;

  enum VGPU_DIRECTION direction;
};


#endif

