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
  VGPU_CUDA_FREE,
  VGPU_CUDA_MEMCPY,
  VGPU_CUDA_REGISTER_FAT_BINARY,
  VGPU_CUDA_REGISTER_FUNCTION,
  VGPU_CUDA_KERNEL_LAUNCH,
  VGPU_CUDA_THREAD_SYNCHRONIZE,
  CMD_MAX,
};

// enum __device_builtin__ cudaMemcpyKind
// {
//     cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
//     cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
//     cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
//     cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
//     cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
// };
enum MemcpyKind
{
    H2H          =   0,      /**< Host   -> Host */
    H2D        =   1,      /**< Host   -> Device */
    D2H        =   2,      /**< Device -> Host */
    D2D      =   3,      /**< Device -> Device */
    cpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

struct VgpuArgs {
  enum VGPU_COMMAND cmd;
  uint64_t src;
  uint64_t src_size;
  uint64_t dst;
  uint64_t dst_size;
  uint64_t flag;

  // return code or pointer
  uint64_t ret;

  uint64_t owner_id;

  enum MemcpyKind kind;
};


#endif

