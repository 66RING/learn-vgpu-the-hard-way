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

#if 1
	#define dprintk(fmt, arg...) printk(fmt, ##arg)
#else
	#define dprintk(fmt, arg...)
#endif

#define panic(msg) printk(msg);abort()
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
	dprintk("send_command");
	struct scatterlist *sgs[2];
	struct scatterlist sg_out, sg_in;

	VgpuArgs *result;
	result = kmalloc(sizeof(VgpuArgs), GFP_KERNEL);
	// TODO:: review and test
	memcpy(result, args, sizeof(VgpuArgs));

	// 创建scatterlist, 与buf绑定
	sg_init_one(&sg_out, args, sizeof(VgpuArgs));
	sg_init_one(&sg_in, result, sizeof(VgpuArgs));
	sgs[0] = &sg_out;
	sgs[1] = &sg_in;

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

// TODO: 假设存在一次数据传输的上限
#define MM_BLOCK_SIZE = 4096;

// size: bytes of data
static uint64_t user_to_gpa(uint64_t src, size_t size) {
	void *gva = kmalloc(size, GFP_KERNEL);
	int err;
	if((err=copy_from_user(gva, (void*)src, size)) != 0) {
		error("copy_from_user failed");
		return 0;
	}
	return (uint64_t)virt_to_phys(gva);
}

// TODO: more other case
static void kfree_gpa(uint64_t gpa, size_t size) {
  kfree(phys_to_virt((phys_addr_t)gpa));
}

static int vgpu_open(struct inode *inode, struct file *filp) {
	dprintk("vgpu_open\n");
	dprintk("dummpy open, nothing to do now.\n");
	// TODO: 注释, 理清楚内核驱动开发中的作用
	try_module_get(THIS_MODULE);

	// TODO:
	return 0;
}

static int vgpu_release(struct inode *inode, struct file *filp) {
	// FIXME: bug 解除占用
	dprintk("vgpu_release\n");
	module_put(THIS_MODULE);
	return 0;
}

static int vgpu_mmap(struct file *filp, struct vm_area_struct *vma) {
	dprintk("vgpu_mmap\n");
	// TODO: useless for now
	return 0;
}

// 考虑用户态数据地址转换问题
static void vgpu_cuda_memcpy(VgpuArgs *arg) {
	dprintk("vgpu_cuda_memcpy\n");
	// TODO: 考虑mmap情况下设备内存和主存的一致性问题
	dprintk("hx: 0x%x, dx 0x%x, size: %d\n", arg->src, arg->dst, arg->src_size);
	switch (arg->kind) {
		case H2H: {
			dprintk("todo vgpu_cuda_memcpy H2H\n");
		} break;
		case H2D: {
			// src: host memory address
			// dst: device memeory address
			dprintk("vgpu_cuda_memcpy H2D\n");
			// 从用户态拷贝数据到设备
			//  1. 获取用户态数据到内核态
			//  2. 包裹转发
			// TODO: 考虑mmap时, 同步更新数据问题

			arg->src = user_to_gpa(arg->src, arg->src_size);
			if (arg->src == 0) {
				return;
			}
			send_command(arg);
			kfree_gpa((uint64_t *)arg->src, arg->src_size);
		} break;
		case D2H: {
			dprintk("vgpu_cuda_memcpy D2H\n");
			// src: device memory address
			// dst: host memeory address
			// 创建内核态缓存
			void* buffer = kmalloc(arg->dst_size, GFP_KERNEL);
			// 计算缓存物理地址
			uint64_t dst_phys = virt_to_phys(buffer);  
			// 传递物理地址给后端
			uint64_t user_dst = arg->dst;
			arg->dst = dst_phys;
			send_command(arg);
			// 数据拷贝回用户态
			copy_to_user((void*)user_dst, buffer, arg->dst_size);
			kfree(buffer);
		} break;
		case D2D: {
			dprintk("todo vgpu_cuda_memcpy D2D\n");
		} break;
		case cpyDefault:
			dprintk("not support direction, UVM only\n");
			break;
		default:
			dprintk("undefine direction of memcpy\n");
			break;
	}
}

// 仅通知后端初始化
// void** __cudaRegisterFatBinary(void *fatCubin);
static void vgpu_cuda_register_fat_binary(VgpuArgs *arg) {
	// 将uva中fatCubin指示的内容转换成物理地址
	send_command(arg);
}

// 加载kernel(function)
static void vgpu_cuda_register_function(VgpuArgs *arg) {
	// 从用户态获取数据, 并转换成物理地址
	arg->src = user_to_gpa(arg->src, arg->src_size);
	arg->dst = user_to_gpa(arg->dst, arg->dst_size);
	
	send_command(arg);
	
	kfree_gpa(arg->src, arg->src_size);
	kfree_gpa(arg->dst, arg->dst_size);
}

// cudaError_t cudaLaunch (const void *func)
// 	args.src: kernel config
// 	args.dst: param config
// 	args.flag: func
static void vgpu_cuda_kernel_launch(VgpuArgs *arg) {
	// 从用户态获取数据, 并转换成物理地址
	arg->src = user_to_gpa(arg->src, arg->src_size);
	arg->dst = user_to_gpa(arg->dst, arg->dst_size);
	
	send_command(arg);
	
	kfree_gpa(arg->src, arg->src_size);
	kfree_gpa(arg->dst, arg->dst_size);
}

static long vgpu_ioctl(struct file *filp, unsigned int _cmd, unsigned long _arg) {
	dprintk("vgpu_ioctl\n");
	VgpuArgs *arg = kmalloc(sizeof(VgpuArgs), GFP_KERNEL);
	int err;
	if((err=copy_from_user(arg, (void*)_arg, sizeof(VgpuArgs))) != 0) {
		dprintk("err copy_from_user");
		return -1;
	}

	dprintk("%d\n", arg->cmd);
	switch (arg->cmd) {
	case VGPU_CUDA_MALLOC:
		send_command(arg);
		break;
	case VGPU_CUDA_FREE:
		send_command(arg);
		break;
	case VGPU_CUDA_REGISTER_FAT_BINARY:
		vgpu_cuda_register_fat_binary(arg);
		break;
	case VGPU_CUDA_REGISTER_FUNCTION:
		vgpu_cuda_register_function(arg);
		break;
	case VGPU_CUDA_MEMCPY:
		vgpu_cuda_memcpy(arg);
		break;
	case VGPU_CUDA_KERNEL_LAUNCH:
		vgpu_cuda_kernel_launch(arg);
		break;
	case VGPU_CUDA_THREAD_SYNCHRONIZE:
		// 简单通知后端调用同步命令
		send_command(arg);
		break;
	default:
		break;
	}

	if((err=copy_to_user((void*)_arg, arg, sizeof(VgpuArgs)))!=0) {
		dprintk("err copy_to_user");
		return -1;
	}

	kfree(arg);
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
	dprintk("virtio_remove\n");
	// 释放驱动
	misc_deregister(&vgpu_driver);
	// 重置设备
	// 释放内存空间
}


static int virtio_probe(struct virtio_device *vdev) {
	dprintk("virtio_probe\n");
	// 创建vgpu全局单例, 负责管理vring等资源
	vdev->priv = vgpu = kzalloc(sizeof(struct virtio_vgpu), GFP_KERNEL);
	vgpu->vdev = vdev;

	// 创建virtqueue, 不绑定 完成回调函数, 一般名为xxx_done
	vgpu->vq = virtio_find_single_vq(vdev, NULL, "command virtqueue");

	// 注册设备驱动
	// 	绑定设备名称, 操作方法, 次设备号
	misc_register(&vgpu_driver);
	// TODO: 初始化锁
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
	dprintk("virtio_vgpu_init done\n");
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

