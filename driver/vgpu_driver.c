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

#define panic(msg) printk(msg);abort()
// TODO: learn this macro
#define error(fmt, arg...)                                                     \
  printk(KERN_ERR "### func= %-30s ,line= %-4d ," fmt, __func__, __LINE__,     \
         ##arg)



// TODO: virtio协议
struct virtio_vgpu {
	// private
	// TODO: 保存备用??
	struct virtio_device *vdev;

	// public
	struct virtqueue *vq;
};

// 全局单例
struct virtio_vgpu* vgpu;

static int vgpu_open(struct inode *inode, struct file *filp);
static int vgpu_release(struct inode *inode, struct file *filp);
static int vgpu_mmap(struct file *filp, struct vm_area_struct *vma);
static long vgpu_ioctl(struct file *filp, unsigned int _cmd, unsigned long _arg);

static struct virtio_device_id id_table[] = {
	// id table中的设备id应与虚拟设备id一致
	{ VIRTIO_ID_VGPU, VIRTIO_DEV_ANY_ID },
    {0},
};


static void send_command(VgpuArgs *args) {
	printk("send_command");
	struct scatterlist *send, *recv, *sgs[2];

	VgpuArgs *result;
	result = kmalloc(sizeof(VgpuArgs), GFP_KERNEL);
	// TODO:: review and test
	memcpy(result, args, sizeof(VgpuArgs));

	// 创建scatterlist, 与buf绑定
	sg_init_one(send, args, sizeof(VgpuArgs));
	sg_init_one(recv, result, sizeof(VgpuArgs));
	sgs[0] = send;
	sgs[1] = recv;

	// TODO: 加锁, 多个程序同时访问驱动

	if(virtqueue_add_sgs(vgpu->vq, sgs, 1, 1, args, GFP_ATOMIC)) {
		error("virtqueue_add_sgs failed");
	}

	// 通知后端virtio事件
	virtqueue_kick(vgpu->vq);

	// 等待后端处理完毕
	int _len;
	while(!virtqueue_get_buf(vgpu->vq, &_len) && !virtqueue_is_broken(vgpu->vq)) {
		// 阻塞则让出cpu
		cpu_relax();
	}

out:
	// 结果返回
	memcpy(args, result, sizeof(VgpuArgs));
	kfree(result);
	// TODO:
}

static int vgpu_open(struct inode *inode, struct file *filp) {
	printk("vgpu_open\n");
	printk("dummpy open, nothing to do now.\n");
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
	VgpuArgs *arg = kmalloc(sizeof(VgpuArgs), GFP_KERNEL);
	int err;
	if((err=copy_from_user(arg, (void*)_arg, sizeof(VgpuArgs))) != 0) {
		printk("err copy_from_user");
		return -1;
	}

	switch (arg->cmd) {
	case VGPU_CUDA_MALLOC:
		break;
	default:
		break;
	}

	if((err=copy_to_user((void*)_arg, arg, sizeof(VgpuArgs)))!=0) {
		printk("err copy_from_user");
		return -1;
	}
	kfree(arg);
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
	// 创建vgpu全局单例, 负责管理vring等资源
	vdev->priv = vgpu = kzalloc(sizeof(struct virtio_vgpu), GFP_KERNEL);
	vgpu->vdev = vdev;

	// 创建virtqueue, 不绑定 完成回调函数, 一般名为xxx_done
	vgpu->vq = virtio_find_single_vq(vdev, NULL, "command virtqueue");

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

