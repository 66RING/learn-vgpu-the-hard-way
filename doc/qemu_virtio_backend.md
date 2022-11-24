---
title: 如何为QEMU写VirtIO设备
author: 66RING
date: 2022-11-12
tags: 
- qemu
- VirtIO
mathjax: true
---

# Abstract

VirtIO前端就是guest中的VirtIO驱动, 对于VirtIO后端的编写, qemu为开发者屏蔽了很多VirtIO协议细节可以方便的实现。

- virtio-pmem比较简单
- handler的入口通过`virtio_add_queue`添加

## VirtIO Transport

VirtIO支持


## QOM

填写`TypeInfo`时, 大多数设备的父类都是`TYPE_DEVICE`, 但是VirtIO设备的父类需要是`TYPE_VirtIO_DEVICE`。

```c
static const TypeInfo VirtIO_blk_info = {
    .name = TYPE_VirtIO_BLK,
    .parent = TYPE_VirtIO_DEVICE,
    .instance_size = sizeof(VirtIOBlock),
    .instance_init = VirtIO_blk_instance_init,
    .class_init = VirtIO_blk_class_init,
};
```

## VirtIOPCIProxy

因为早期QOM系统的遗留问题, PCI VirtIO设备和正常设备层次结构是不同的: 设备需要基于`VirtIOPCIProxy`类并且需要手动实例化VirtIO实例(而不是QOM中的自动父类初始化, 因为他们在两条分支上了)。

```c
/*
 * virtio-blk-pci: This extends VirtioPCIProxy.
 */
#define TYPE_VIRTIO_BLK_PCI "virtio-blk-pci-base"
DECLARE_INSTANCE_CHECKER(VirtIOBlkPCI, VIRTIO_BLK_PCI,
                         TYPE_VIRTIO_BLK_PCI)

struct VirtIOBlkPCI {
    VirtIOPCIProxy parent_obj;  // virtio-pci类, 继承pci-device
    VirtIOBlock vdev;           // virtio-blk类, 继承virtio-device
};

static Property virtio_blk_pci_properties[] = {
    DEFINE_PROP_UINT32("class", VirtIOPCIProxy, class_code, 0),
    DEFINE_PROP_BIT("ioeventfd", VirtIOPCIProxy, flags,
                    VIRTIO_PCI_FLAG_USE_IOEVENTFD_BIT, true),
    DEFINE_PROP_UINT32("vectors", VirtIOPCIProxy, nvectors,
                       DEV_NVECTORS_UNSPECIFIED),
    DEFINE_PROP_END_OF_LIST(),
};

static void virtio_blk_pci_realize(VirtIOPCIProxy *vpci_dev, Error **errp)
{
    VirtIOBlkPCI *dev = VIRTIO_BLK_PCI(vpci_dev);
    DeviceState *vdev = DEVICE(&dev->vdev);

    ...

    qdev_realize(vdev, BUS(&vpci_dev->bus), errp);
}

static void virtio_blk_pci_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtioPCIClass *k = VIRTIO_PCI_CLASS(klass);
    PCIDeviceClass *pcidev_k = PCI_DEVICE_CLASS(klass);

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);
    device_class_set_props(dc, virtio_blk_pci_properties);
    k->realize = virtio_blk_pci_realize;
    pcidev_k->vendor_id = PCI_VENDOR_ID_REDHAT_QUMRANET;
    pcidev_k->device_id = PCI_DEVICE_ID_VIRTIO_BLOCK;
    pcidev_k->revision = VIRTIO_PCI_ABI_VERSION;
    pcidev_k->class_id = PCI_CLASS_STORAGE_SCSI;
}

static void virtio_blk_pci_instance_init(Object *obj)
{
    VirtIOBlkPCI *dev = VIRTIO_BLK_PCI(obj);

    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev),
                                TYPE_VIRTIO_BLK);
    object_property_add_alias(obj, "bootindex", OBJECT(&dev->vdev),
                              "bootindex");
}

static const VirtioPCIDeviceTypeInfo virtio_blk_pci_info = {
    .base_name              = TYPE_VIRTIO_BLK_PCI,
    .generic_name           = "virtio-blk-pci",
    .transitional_name      = "virtio-blk-pci-transitional",
    .non_transitional_name  = "virtio-blk-pci-non-transitional",
    .instance_size = sizeof(VirtIOBlkPCI),
    .instance_init = virtio_blk_pci_instance_init,
    .class_init    = virtio_blk_pci_class_init,
};
```

可见需要手动实例化底层`TYPE_VIRTIO_BLK`, 并且添加到对应PCI设备`VIRTIO_BLK_PCI(obj)`。

```
virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev),
                            TYPE_VIRTIO_BLK);
object_property_add_alias(obj, "bootindex", OBJECT(&dev->vdev),
                          "bootindex");
```

一般写在一个独立的`virtio-xxx-pci.c`文件, 模式固定


## Back End Implementations

可以在以下几个地方实现VirtIO后端

- qemu
- host内核
- 一个单独的进程



# VirtIO驱动

- VirtIO驱动和设备通过设备id来匹配


# VirtIO设备

## Ref

- https://www.qemu.org/docs/master/devel/virtio-backends.html

