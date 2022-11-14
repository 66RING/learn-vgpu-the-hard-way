#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu-common.h"
#include "qemu/error-report.h"
#include "qemu/main-loop.h"
#include "hw/virtio/virtio-vgpu.h"
#include "hw/qdev-properties.h"
#include "sysemu/hostmem.h"
#include "block/aio.h"
#include "block/thread-pool.h"
#include "trace.h"
#include "standard-headers/linux/virtio_ids.h"


static void virtio_vgpu_handler(VirtIODevice *vdev, VirtQueue *vq)
{
	panic("unimplament virtio_vgpu_handler");
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
