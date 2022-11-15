#include <linux/err.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/random.h>
#include <linux/scatterlist.h>
#include <linux/sort.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/virtio.h>
#include <linux/virtio_ids.h>
#include <linux/virtio_pci.h>

#include <linux/virtio_config.h>

#include "vgpu_driver.h"

// TODO: virtio协议
struct virtio_vgpu {
	// private
	// TODO: 保存备用
	struct virtio_device *vdev;

	// public
	struct virtqueue *vq;
};

static int vgpu_open(struct inode *inode, struct file *filp);
static int vgpu_release(struct inode *inode, struct file *filp);
static int vgpu_mmap(struct file *filp, struct vm_area_struct *vma);
static long vgpu_ioctl(struct file *filp, unsigned int _cmd, unsigned long _arg);

static struct virtio_device_id id_table[] = {
	// id table中的设备id应与虚拟设备id一致
	{ VIRTIO_ID_VGPU, VIRTIO_DEV_ANY_ID },
    {0},
};

static int vgpu_open(struct inode *inode, struct file *filp) {
	printk("vgpu_open\n");
	// TODO:
	return 0;
}

static int vgpu_release(struct inode *inode, struct file *filp) {
	printk("vgpu_release\n");
	// TODO:
	return 0;
}

static int vgpu_mmap(struct file *filp, struct vm_area_struct *vma) {
	printk("vgpu_mmap\n");
	// TODO:
	return 0;
}

static long vgpu_ioctl(struct file *filp, unsigned int _cmd, unsigned long _arg) {
	printk("vgpu_ioctl\n");
	// TODO:
	return 0;
}

// 说创建的设备文件将通过这里绑定的方法进行操作
static struct file_operations vgpu_fops = {
    .owner = THIS_MODULE,
    .open = vgpu_open,
    .release = vgpu_release,
    .unlocked_ioctl = vgpu_ioctl,
    .mmap = vgpu_mmap,
};

static struct miscdevice vgpu_driver = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "vgpu",
	.fops = &vgpu_fops,
};

static void virtio_remove(struct virtio_device *vdev) {
	// TODO:
	printk("virtio_remove\n");
	// 释放驱动
	misc_deregister(&vgpu_driver);
	// 重置设备
	// 释放内存空间
}

static int virtio_probe(struct virtio_device *vdev) {
	printk("virtio_probe\n");
	struct virtio_vgpu* vgpu;

	// TODO:
	vdev->priv = vgpu = kzalloc(sizeof(struct virtio_vgpu), GFP_KERNEL);
	vgpu->vdev = vdev;

	// 创建virtqueue
	// // 暂时不需要
	// vgpu->vq = virtio_find_single_vq(vdev, , "command virtqueue");

	// 注册设备驱动
	// 	绑定设备名称, 操作方法, 次设备号
	misc_register(&vgpu_driver);
	// 初始化锁
	return 0;
}

static unsigned int features[] = {};

static struct virtio_driver virtio_driver = {
	.driver.owner = THIS_MODULE,
    .driver.name = "vgpu",
    .id_table = id_table,
    .probe = virtio_probe,
    .remove = virtio_remove,
};

// 内核模块初始化, 注册驱动构造方法
static int virtio_vgpu_init(void)
{
    if (register_virtio_driver(&virtio_driver) < 0) {
		return 1;
	}
	printk("virtio_vgpu_init done\n");
	return 0;
}

// 移除驱动
static void virtio_vgpu_exit(void)
{
	unregister_virtio_driver(&virtio_driver);
}

module_init(virtio_vgpu_init);
module_exit(virtio_vgpu_exit);

MODULE_DEVICE_TABLE(virtio, id_table);
MODULE_DESCRIPTION("Vgpu Virtio driver");
MODULE_LICENSE("GPL");
MODULE_AUTHOR("66RING");

