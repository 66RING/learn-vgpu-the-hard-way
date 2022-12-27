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
static inline void __cudaErrorCheck(CUresult err, const int line) {
    char *str;
    if (err != cudaSuccess) {
		cuGetErrorName(err, (const char **) &str);
        error_log("[CUDA error] %04d \"%s\" line %d\n", err, str, line);
    }
}


/*
 * global and type
 */
#define MaxFunctionNum 100
#define MaxDeviceNum 10
#define MaxStreamNum 512

// kernel -> function 一一对应
typedef struct deviceKernel {
	void* fatBin;
	char funcName[100];
	uint32_t funcId;
} deviceKernel;

// 记录所有所需kernel备用
// TODO: 小心reload, 如果线程切换了就load null了
struct {
	deviceKernel kernelLoaded[MaxFunctionNum];
	uint32_t size;
} kernelList;

static deviceKernel* kernelNew() {
	if(kernelList.size >= MaxFunctionNum) {
		debug_log("run out of kernel slot\n");
		return NULL;
	}

	kernelList.kernelLoaded[kernelList.size].fatBin = malloc(4*1024*1024);
	return &kernelList.kernelLoaded[kernelList.size++];
}


// table of function handle
typedef struct functionHandle {
	uint32_t key;
	CUfunction cudaFunction;
} functionHandle;

typedef struct cudaDevice {
	CUdevice device;
	CUcontext context;
	functionHandle funcTable[MaxFunctionNum];
	uint32_t size;
} cudaDevice;

static CUfunction* cudaFuncNew(cudaDevice *dev, uint32_t key) {
	if (dev->size >= MaxFunctionNum) {
		debug_log("out of function slot");
		return NULL;
	}
	functionHandle *handle = &dev->funcTable[dev->size];
	dev->size++;
	handle->key = key;
	return &handle->cudaFunction;
}

static CUfunction* cudaFuncGet(cudaDevice *dev, uint32_t key) {
	int i;
	for (i=0;i<dev->size;i++) {
		if (dev->funcTable[i].key == key)
			break;
	}

	if(i == dev->size) {
		debug_log("function not found with key: %d", key);
		return NULL;
	}
	return &dev->funcTable[i].cudaFunction;
}

// thread id -> 找到对应的device, 因为它的上下文在哪里
// TODO: 目前只考虑一个设备, 总是device 0
typedef struct cudaRuntime {
	cudaDevice pool[10];
} cudaRuntime;

cudaStream_t cudaStream[MaxStreamNum];
cudaRuntime cudaPool;
int deviceCount = -1;


/*
 * cuda helper function
 */
static void cudaInit() {
	// 初始cuda driver api
	cudaErrorCheck(cuInit(0));
	// TODO: 目前只考虑一个设备的情况, 只对一个初始化
	cudaErrorCheck(cuDeviceGetCount(&deviceCount));
	debug_log( "cuda device count: %d\n", deviceCount);

	// 创建设备上下文
	cudaErrorCheck(cuDeviceGet(&cudaPool.pool[0].device, 0));
	cudaErrorCheck(cuCtxCreate(&cudaPool.pool[0].context, 0, cudaPool.pool[0].device));

	// 创建default stream
	cudaStreamCreate(&cudaStream[0]);
}

static void loadKernel(int devId, uint32_t key, deviceKernel* kernel) {
	CUmodule module;
	// 从fatbin中加载module到当前context, 返回module
	cudaErrorCheck(cuModuleLoadData(&module, kernel->fatBin));
	// 从module中加载函数, 返回一个funcHandle
	CUfunction *func = cudaFuncNew(&cudaPool.pool[devId], key);
    cudaErrorCheck(cuModuleGetFunction(func, module, kernel->funcName));
}

/*
 * backend logic
 */
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

	CUresult err = CUDA_ERROR_UNKNOWN;
	void *devPtr;
	cudaErrorCheck(err = (CUresult)cudaMalloc(&devPtr, args->dst_size));
	args->dst = (uint64_t)devPtr;
	args->ret = err;
    debug_log("< vgpu_cuda_malloc: 0x%lx\n", args->dst);
}

static void vgpu_cuda_free(VgpuArgs* args) {
    debug_log("> vgpu_cuda_free\n");
	CUresult err = CUDA_ERROR_UNKNOWN;
	err = (CUresult)cudaFree((void*)args->dst);
	args->ret = err;
    debug_log("< vgpu_cuda_free: 0x%lx\n", args->dst);
}

// gpu内存拷贝
//  cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
//  cudaMemcpyHostToHost
//  cudaMemcpyHostToDevice
//  cudaMemcpyDeviceToHost
//  cudaMemcpyDeviceToDevice
//  cudaMemcpyDefault, 仅在支持uvm的gpu中适用
//
static void vgpu_cuda_memcpy(VgpuArgs* args) {
    debug_log("> vgpu_cuda_memcpy\n");
    debug_log("vgpu_cuda_memcpy src: 0x%lx, dst 0x%lx, size: %ld, kind: %d\n",
			args->src, args->dst, args->src_size, args->kind);

	uint64_t* src_hva;
	uint64_t* dst_hva;
	CUresult err = CUDA_ERROR_UNKNOWN;

	switch (args->kind) {
		case cudaMemcpyHostToHost: {
			panic("unimplament");
		} break;
		case cudaMemcpyHostToDevice: {
			if ((src_hva = gpa2hva(args->src)) == NULL) {
				panic("gpa2hva failed");
			}

			DEBUG_BLOCK(
				debug_log("H2D host data: ");
				inspect(src_hva, args->src_size, uint8_t);
			);
			// TODO: 待真实gpu测试driver API
			cudaErrorCheck(err = (CUresult)cudaMemcpy(args->dst, src_hva, args->src_size, H2D));
			// cudaErrorCheck(err = cuMemcpyHtoD((CUdeviceptr)args->dst, src_hva, args->src_size));
		} break;
		case cudaMemcpyDeviceToHost: {
			debug_log("D2H: ");
			if ((dst_hva = gpa2hva(args->dst)) == NULL) {
				panic("gpa2hva failed");
			}
			// TODO: 待真实gpu测试driver API
			cudaErrorCheck(err = (CUresult)cudaMemcpy(dst_hva, args->src, args->dst_size, D2H));
			// cudaErrorCheck(err = cuMemcpyDtoH((CUdeviceptr)dst_hva, args->src, args->dst_size));

			DEBUG_BLOCK(
				debug_log("D2H device data: ");
				inspect(dst_hva, args->dst_size, uint8_t);
			);
		} break;
		case cudaMemcpyDeviceToDevice: {
			panic("unimplament");
		} break;

		case cudaMemcpyDefault:
			panic("not support direction, UVM only");
			break;
		default:
			debug_log("undefine kind %d\n", args->kind);
			// panic("undefine direction of memcpy");
			break;
	}

	args->ret = err;
    debug_log("< vgpu_cuda_memcpy: \n");
}

// void** __cudaRegisterFatBinary(void *fatCubin)
static void vgpu_cuda_register_fat_binary(VgpuArgs* args) {
    debug_log("> vgpu_cuda_register_fat_binary: \n");
	cudaInit();
    debug_log("< vgpu_cuda_register_fat_binary: \n");
}

// kernel < -- > function 一一对应, 加载function就是加载kernel
static void vgpu_cuda_register_function(VgpuArgs* args) {
	void *fatBin;
	char *funcName;
	uint32_t funcId;

	// TODO: 暂不考虑fatbin很大的情况
    fatBin = gpa2hva(args->src);
    funcName = gpa2hva(args->dst);
    funcId = args->flag;

	// 初始化kernel上下文
	// 假设fatBin不大于4MB
	deviceKernel *ptr = kernelNew();
    memcpy(ptr->fatBin, fatBin, args->src_size);
    memcpy(ptr->funcName, funcName, args->dst_size);
    ptr->funcId = funcId;

	// TODO: 目前仅考虑一个设备
	loadKernel(0, funcId, ptr);


}

static void virtio_vgpu_handler(VirtIODevice *vdev, VirtQueue *vq)
{
	VgpuArgs *args;
	VirtQueueElement *elem;

	args = (VgpuArgs*)malloc(sizeof(VgpuArgs));
	// virtio协议, 传入N个入M处的scatterlist, 可以从iovector中获取到
	while((elem = virtqueue_pop(vq, sizeof(VirtQueueElement)))) {
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
		case VGPU_CUDA_REGISTER_FAT_BINARY:
			vgpu_cuda_register_fat_binary(args);
			break;
		case VGPU_CUDA_REGISTER_FUNCTION:
			vgpu_cuda_register_function(args);
			break;
		default:
			panic("unknow command");
			break;
		}

		// 数据更新后写回iovector
		size_t s = iov_from_buf(elem->in_sg, elem->in_num, 0, args, sizeof(VgpuArgs));
		if (unlikely(s != sizeof(VgpuArgs))) {
			debug_log("iov_from_buf size error \n");
		}

		// 将处理完后的描述符索引更新到Used队列中
		virtqueue_push(vq, elem, sizeof(VgpuArgs));
		// 通知前端
		virtio_notify(vdev, vq);
	}

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

	// TODO: review
	// vgpu运行时相关初始化
	kernelList.size = 0;
	for (int i=0;i<10;i++) {
		cudaPool.pool[i].size = 0;
	}
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
