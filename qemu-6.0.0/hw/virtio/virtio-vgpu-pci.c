#include "qemu/osdep.h"

#include "hw/qdev-properties.h"
#include "hw/virtio/virtio-vgpu.h"
#include "virtio-pci.h"
#include "qapi/error.h"
#include "qemu/module.h"
#include "qom/object.h"

typedef struct VirtIOVgpuPCI VirtIOVgpuPCI;

/*
 * virtio-vgpu-pci: This extends VirtioPCIProxy.
 */
#define TYPE_VIRTIO_VGPU_PCI "virtio-vgpu-pci-base"
DECLARE_INSTANCE_CHECKER(VirtIOVgpuPCI, VIRTIO_VGPU_PCI,
                         TYPE_VIRTIO_VGPU_PCI)

struct VirtIOVgpuPCI {
    VirtIOPCIProxy parent_obj;
    VirtIOVGPU vdev;
};

// TODO: 功能补充
static Property virtio_vgpu_pci_properties[] = {
    DEFINE_PROP_END_OF_LIST(),
};

// 初始化设备对应pci设备本体
static void virtio_vgpu_pci_realize(VirtIOPCIProxy *vpci_dev, Error **errp)
{
    VirtIOVgpuPCI *dev = VIRTIO_VGPU_PCI(vpci_dev);
    DeviceState *vdev = DEVICE(&dev->vdev);

    qdev_realize(vdev, BUS(&vpci_dev->bus), errp);
}

// pci设备类元数据初始化
static void virtio_vgpu_pci_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pcidev_k = PCI_DEVICE_CLASS(klass);
    VirtioPCIClass *k = VIRTIO_PCI_CLASS(klass);

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
    device_class_set_props(dc, virtio_vgpu_pci_properties);

	// 初始化设备构造函数
    k->realize = virtio_vgpu_pci_realize;
	// 初始化设备信息
    pcidev_k->vendor_id = PCI_VENDOR_ID_REDHAT_QUMRANET;
    pcidev_k->device_id = PCI_DEVICE_ID_VIRTIO_VGPU;
	pcidev_k->revision = VIRTIO_PCI_ABI_VERSION;
   	pcidev_k->class_id  = PCI_CLASS_OTHERS;
}

static void virtio_vgpu_pci_instance_init(Object *obj)
{
    VirtIOVgpuPCI *dev = VIRTIO_VGPU_PCI(obj);

	// 初始化virtio-vgpu设备: TYPE_VIRTIO_VGPU
    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev),
                                TYPE_VIRTIO_VGPU);
}

// 向QOM注册设备对象
static const VirtioPCIDeviceTypeInfo virtio_vgpu_pci_info = {
    .base_name              = TYPE_VIRTIO_VGPU_PCI,
    .generic_name           = "virtio-vgpu-pci",
    .instance_size          = sizeof(VirtIOVgpuPCI),
    .instance_init          = virtio_vgpu_pci_instance_init,
    .class_init             = virtio_vgpu_pci_class_init,
};

static void virtio_vgpu_pci_register(void)
{
    virtio_pci_types_register(&virtio_vgpu_pci_info);
}

type_init(virtio_vgpu_pci_register)

