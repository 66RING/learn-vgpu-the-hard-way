#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu-common.h"
#include "qemu/error-report.h"
#include "qemu/main-loop.h"
#include "qemu/iov.h"
#include "hw/virtio/virtio-vgpu.h"
#include "hw/qdev-properties.h"
#include "sysemu/hostmem.h"
#include "block/aio.h"
#include "block/thread-pool.h"
#include "trace.h"
#include "standard-headers/linux/virtio_ids.h"
#include "qemu/log.h"
#include "exec/address-spaces.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_profiler_api.h>

/*
 * Utils
 */
#if 1
	#define debug_log(fmt, arg...) qemu_log_mask(LOG_GUEST_ERROR, fmt, ##arg);
#else
	#define debug_log(fmt, arg...)
#endif

#if 1
	#define DEBUG_BLOCK(x) do { x }  while(0)
#else
	#define DEBUG_BLOCK(x) 
#endif 

#define error_log(fmt, arg...) qemu_log_mask(LOG_GUEST_ERROR, fmt, ##arg);

#define inspect(src_ptr, len, type) \
	do { \
		int ___i; \
		for (___i=0;___i<len;___i++) { \
			qemu_log_mask(LOG_GUEST_ERROR, "0x%x ", ((type*)src_ptr)[___i]); \
		} \
		qemu_log_mask(LOG_GUEST_ERROR, "\n"); \
	} while(0)


#define cudaErrorCheck(err) __cudaErrorCheck(err, __LINE__)
static inline void __cudaErrorCheck(cudaError_t err, const int line) {
    char *str;
    if (err != cudaSuccess) {
        str = (char *) cudaGetErrorString(err);
        error_log(LOG_GUEST_ERROR, "[CUDA error] %04d \"%s\" line %d\n", err, str, line);
    }
}


/*
 * global and type
 */
typedef struct cudaDevice {
	CUdevice device;
	CUcontext context;

} cudaDevice;

// TODO: 先一个
cudaDevice devicePool;
int deviceCount = -1;


/*
 * Declaration
 */
static void cudaInit() {
	cudaErrorCheck(cuInit(0));
	cudaErrorCheck(cuDeviceGetCount(&deviceCount));
	debug_log( "cuda device count: %d\n", deviceCount);

	// TODO: 注意create是一个栈, 后进先出
	cudaErrorCheck(cuDeviceGet(&devicePool.device, 0));
	cudaErrorCheck(cuCtxCreate(&devicePool.context, 0, devicePool.device));
}

static void* gpa2hva(uint64_t gpa) {
	MemoryRegionSection section;

	// TODO: about size = 1
	section = memory_region_find(get_system_memory(), (ram_addr_t)gpa, 1);
	if (!section.mr) {
		error_log("mr not found\n");
		return NULL;
	}

	if (!memory_region_is_ram(section.mr)) {
		error_log("gpa2hva not a ram\n");
		memory_region_unref(section.mr);
		return NULL;
	}

	return (memory_region_get_ram_ptr(section.mr) + section.offset_within_region);
}

// 转发cudaError_t cudaMalloc(void **devPtr, size_t size);
// 申请gpu设备内存
static void vgpu_cuda_malloc(VgpuArgs* args) {
    debug_log("> vgpu_cuda_malloc\n");

	void *devPtr;
	cudaErrorCheck(cudaMalloc(&devPtr, args->dst_size));
	args->dst = (uint64_t)devPtr;
    debug_log("< vgpu_cuda_malloc: 0x%lx\n", args->dst);
	// TODO:返回错误代码
}

static void vgpu_cuda_free(VgpuArgs* args) {
    debug_log("> vgpu_cuda_free\n");
	cudaFree((void*)args->dst);
    debug_log("< vgpu_cuda_free: 0x%lx\n", args->dst);
	// TODO:返回错误代码
}

// gpu内存拷贝
//  cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
//  cudaMemcpyHostToHost
//  cudaMemcpyHostToDevice
//  cudaMemcpyDeviceToHost
//  cudaMemcpyDeviceToDevice
//  cudaMemcpyDefault, 仅在支持uvm的gpu中适用
//
// TODO: error handling
static void vgpu_cuda_memcpy(VgpuArgs* args) {
    debug_log("> vgpu_cuda_memcpy\n");
    debug_log("vgpu_cuda_memcpy src: 0x%lx, dst 0x%lx, size: %d\n", args->src, args->dst, args->src_size);

	uint64_t* src_hva;
	uint64_t* dst_hva;
	CUresult err = 0;

	switch (args->kind) {
		case H2H: {
			panic("unimplament");
		} break;
		case H2D: {
			if ((src_hva = gpa2hva(args->src)) == NULL) {
				panic("gpa2hva failed");
			}

			DEBUG_BLOCK(
				debug_log("H2D host data: ");
				inspect(src_hva, args->src_size, uint8_t);
			);
			// TODO: 待真实gpu测试driver API
			cudaErrorCheck(cudaMemcpy(args->dst, src_hva, args->src_size, H2D));
			// cudaErrorCheck(err = cuMemcpyHtoD((CUdeviceptr)args->dst, src_hva, args->src_size));
		} break;
		case D2H: {
			debug_log("D2H: ");
			if ((dst_hva = gpa2hva(args->dst)) == NULL) {
				panic("gpa2hva failed");
			}
			// TODO: 待真实gpu测试driver API
			cudaErrorCheck(cudaMemcpy(dst_hva, args->src, args->dst_size, D2H));
			// cudaErrorCheck(err = cuMemcpyDtoH((CUdeviceptr)dst_hva, args->src, args->dst_size));

			DEBUG_BLOCK(
				debug_log("D2H device data: ");
				inspect(dst_hva, args->dst_size, uint8_t);
			);
		} break;
		case D2D: {
			panic("unimplament");
		} break;

		case cpyDefault:
			panic("not support direction, UVM only");
			break;
		default:
			panic("undefine direction of memcpy");
			break;
	}

    debug_log("< vgpu_cuda_memcpy: \n");
}

static void virtio_vgpu_handler(VirtIODevice *vdev, VirtQueue *vq)
{
	VgpuArgs *args;
	VirtQueueElement *elem;

	args = (VgpuArgs*)malloc(sizeof(VgpuArgs));
	// virtio协议, 传入N个入M处的scatterlist, 可以从iovector中获取到
	while(elem = virtqueue_pop(vq, sizeof(VirtQueueElement))) {
		// WARN: error handling
		// 从iovector中读取数据
		iov_to_buf(elem->out_sg, elem->out_num, 0, args, sizeof(VgpuArgs));

		// 读取数据后清除已经读出的数据
		//  对iov及其长度进行修改
		iov_discard_front(&elem->out_sg, &elem->out_num, sizeof(VgpuArgs));

		// 根据vgpu协议执行响应的操作
		switch (args->cmd){
		case VGPU_CUDA_MALLOC:
			vgpu_cuda_malloc(args);
			break;
		case VGPU_CUDA_FREE:
			vgpu_cuda_free(args);
			break;
		case VGPU_CUDA_MEMCPY:
			vgpu_cuda_memcpy(args);
			break;
		default:
			panic("unknow command");
			break;
		}

		// 数据更新后写回iovector
		size_t s = iov_from_buf(elem->in_sg, elem->in_num, 0, args, sizeof(VgpuArgs));
		// TODO: error handling

		// 将处理完后的描述符索引更新到Used队列中
		virtqueue_push(vq, elem, sizeof(VgpuArgs));
		// 通知前端
		virtio_notify(vdev, vq);
	}

out:
	free(args);
}

static uint64_t virtio_vgpu_get_features(VirtIODevice *vdev, uint64_t features,
                                        Error **errp)
{
    return features;
}

// virito设备构造函数
// 监听virtio queue
static void virtio_vgpu_realize(DeviceState *dev, Error **errp)
{
	VirtIODevice *vdev = VIRTIO_DEVICE(dev);
	VirtIOVGPU *vgpu = VIRTIO_VGPU(dev);

    virtio_init(vdev, TYPE_VIRTIO_VGPU, VIRTIO_ID_VGPU,
                sizeof(struct VirtIOVgpuConf));
    vgpu->vq = virtio_add_queue(vdev, 128, virtio_vgpu_handler);
}

static void virtio_vgpu_unrealize(DeviceState *dev)
{
	panic("unimplament virtio_vgpu_unrealize");
}

// 添加用户可定义参数mem_size
static Property virtio_vgpu_properties[] = {
	DEFINE_PROP_SIZE("mem_size", VirtIOVGPU, conf.mem_size, 0),
    DEFINE_PROP_END_OF_LIST(),
};


static void virtio_vgpu_instance_init(Object *obj)
{
	// 暂不需要实现
}

// 初始化虚拟设备类的元数据, e.g. 成员方法realize, 可以自定义的属性等
static void virtio_vgpu_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

    device_class_set_props(dc, virtio_vgpu_properties);

    vdc->realize = virtio_vgpu_realize;
    vdc->unrealize = virtio_vgpu_unrealize;
	vdc->get_features = virtio_vgpu_get_features;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

// 使用QOM框架注册虚拟设备
static TypeInfo virtio_vgpu_info = {
    .name          = TYPE_VIRTIO_VGPU,
    .parent        = TYPE_VIRTIO_DEVICE,
    .class_init    = virtio_vgpu_class_init,
	.instance_init = virtio_vgpu_instance_init,
    .instance_size = sizeof(VirtIOVGPU),
};

static void virtio_register_types(void)
{
    type_register_static(&virtio_vgpu_info);
}

type_init(virtio_register_types)
