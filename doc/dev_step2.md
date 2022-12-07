---
title: step2
author: 66RING
date: 2022-11-20
tags: 
- gpu
- qemu
mathjax: true
---

# Abstract


# Preface


# Overview

# 第二阶段开发: API试个遍

## qemu虚拟设备

TODO: 理清virtio虚拟设备创建流程, virtio pci等

设备realize的时候, 使用`virtio_add_queue`创建virtqueue, 并绑定virtqueue的处理函数

- virtqueue handler
    * `virtqueue_pop`从avail ring中获取数据: 事件元素抽象`VirtQueueElement* elem`
    * 然后使用`iov_to_buf`从`elem->out_sg`中获取参数, 使用`iov_discard_front`配合删除已用
    * 数据获取完毕(相当于数据拷贝到内核空间, 这里就是从virtqueue空间拷贝出来), 根据vgpu协议执行对应功能
    * `iov_from_buf`写回iov, 我们`virtqueue_add_sgs`传入的时候指定了有几个入出
    * `virtqueue_push`将avail ring取出的数据(elem)放回到used ring
    * `virtio_notify`通知前端
    * 释放资源, buffer
    * API
        + `virtqueue_pop`
            + 从avail ring中取出数据
        + `iov_to_buf`
            + 将数据从"virtqueue空间"拷贝到"handler空间"
            + 直接从iovec中读取数据`iov_to_buf(iov, iovcnt, offset, buf, bytes);
        + `iov_from_buf`
            + 写回`iov`
        + `iov_discard_front`
            + 从vec前端移除bytes个数据, 返回实际移除的数量, 传入iov和iovcnt更新`iov_discard_front(struct iovec **iov, unsigned int *iov_cnt, size_t bytes)`
        + `virtqueue_push`
            + 写回到used ring
        + `virtio_notify`
            + 通知guest驱动


## virtio内核驱动

- open
    * TODO: `try_module_get(THIS_MODULE)`
        + TODO: 测试是否必要
    * 创建创建设备内存, 即`filp->private`
        + 链表模拟设备上的内存, 后续会实现D2H, H2D的数据拷贝
            + 内核链表的创建可以使用`INIT_LIST_HEAD(&head)`宏来初始化`struct list_head`
    * 约定链表每个block的大小
    * 向vgpu设备发送启动命令
- release
    * `kfree`释放open中申请的资源(设备内存)`filp->private`
    * `module_put(THIS_MODULE)`
- ioctl
    * 申请内核buffer读入用户态数据
        + `copy_from_user`, 从用户态拷贝数据(`args`)到内核态
    * swtich做响应的功能
    * 内核buffer数据拷贝回用户态, `copy_to_user`
    * 释放资源(内核buffer)
    * CUDA API抽象
        + 使用dst, src等明显意义的名字
        + 还是使用ptr1, ptr2等位置标记的名字
- command send
    * `sg_init_one(&new_sg, buf, size)`创建若干个scatterlist, N个出M个入, 这里只需要一入一出
        + 申请buffer，与待创建的sg绑定
        + `virtqueue_add_sgs`向virtqueue传递scatterlist
    * `virtqueue_add_sgs`将scatterlist添加到vring中, 也是在内部转换整buf和desc的添加
        + linux/drivers/virtio/virtio_ring.c
    * `virtqueue_kick`通知后端
    * `while virtqueue_get_buf()`等待后端处理完成
        + `virtqueue_is_broken`而外检测队列没有关闭
    * API
        + `sg_init_one(&new_sg, buf, size)`
            + 新建scatterlist: `new_sg`, 与buf绑定
        + `virtqueue_add_sgs`
            + scatterlist添加到vring中
            + 传入的是scatterlist的数组，参数指定数组中多少个做输入, 多少个做输出
        + `virtqueue_get_buf`
            + get next used buffer, 可以用户检测后端是否处理完成
            + 后端处理完成后会添加到used ring中, 所以可以使用`while`加`cpu_relax`的方式忙等
        + `virtqueue_is_broken`
            + 是否已经关闭
        + `virtqueue_kick`通知后端
- mmap
    * 暂不实现
- `virt_to_phys`


## 用户库

因为可能存在多个线程使用gpu, 需要对齐进行区分。可以使用`syscall(__NR_gettid)`获取线程id作为区分依据。


## CUDA参数模式

如何查看cuda程序需要哪些API: 使用nm命令查看编译好的二进制文件的符号表, 然后查看[CUDA toolkit官网API文档](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)实现具体warpper库。

- 资源分配类一般是可以抽象成`src`和`dst`的
- 或者可直接根据参数位置设置, 类似qemu中`opaque`的设计


### 最简单的CUDA程序

我们这里的最简CUDA程序需要用到如下API, 这些函数调用会被nvcc进一步编译成更多的内部函数调用

- `cudaMalloc`
- `cudaMemcpy`
- `cudaThreadSynchronize`
- `cudaFree`

常见的传统cuda程序流程如下:

1. 申请host内存(系统内存)和申请device内存(gpu显存: `cudaMalloc`), e.g. 动态数组
2. host端数据初始化完毕后使用`cudaMemcpy`拷贝到设备
3. cuda kernel开始执行, 即`<<<>>>`特殊标记的那个函数
4. 必要时使用`cudaThreadSynchronize()`等待gpu执行完成
5. 获取结果: `cudaMemcpy`从显存拷贝回host

程序经过cuda编译器编译后, cuda kernel调用将会被编译成若干函数调用。比如编译成`pushConfigration`, 和`kernelLaunch`等。

这里选较低版本的cuda(cuda9.0, 但是官方已经不再维护, 可以使用这个[备用链接](https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8))是因为它内部细节会简单一点, 编译后的产物比较直观, 资料也比较多。

以下是使用`nm`工具得到的符号表的使用情况, (注意使用`nvcc --cudart shared`编译, 这样能清晰得到对cuda库的使用情况), 以下是我们需要实现的API类型, 可以看到经过nvcc编译后多出了很多函数调用。

```
$ nvcc nvcc --cudart shared -o test test.cu
$ nm ./test | grep libcudart
                 U cudaConfigureCall@@libcudart.so.9.0
                 U cudaFree@@libcudart.so.9.0
                 U __cudaInitModule@@libcudart.so.9.0
                 U cudaLaunch@@libcudart.so.9.0
                 U cudaMalloc@@libcudart.so.9.0
                 U cudaMemcpy@@libcudart.so.9.0
                 U __cudaRegisterFatBinary@@libcudart.so.9.0
                 U __cudaRegisterFunction@@libcudart.so.9.0
                 U cudaSetupArgument@@libcudart.so.9.0
                 U cudaThreadSynchronize@@libcudart.so.9.0
                 U __cudaUnregisterFatBinary@@libcudart.so.9.0
```

要想实现这些API需要四处收集资料, 因为部分API英伟达是不公开的, 而且有的API在最新的文档中是删除了的, 所以新闻档中没有可以看看旧文档。

- [CUDA RUNTIME API v5.5](https://cs.colby.edu/courses/S16/cs336/online_materials/CUDA_Runtime_API.pdf)


## kernel API

- `kmalloc`
- `INIT_LIST_HEAD(struct list_head)`初始化链表结构
- `scatterlist`
    * 将分散的物理内存以list的性质组织起来
    * page为单位
    * 应用场景:
        + DMA传输时只能以连续物理内存为单位进行, 而大多数情况我们是虚拟内存连续和物理内存不连续, 所以scatterlist就是将不连续的物理内存组织起来
    * `sg_init_one`
    * virtio使用scatterlist传递buffer, N个入M个出, [The order is fixed (out followed by in)](https://lwn.net/Articles/239238)


## VirtIO流程

virtio基本流程抽象:

```
Driver端(Guest)发送: 从avail ring中取, 然后通知, host端pop取出后写回used ring中, 通知Guest
Device端(Hose)发送(e.g. 外来信息): 从avail ring中取, 处理完成后写回used ring, 通知Guest

               Host    |     Guest

                      kick
  2.virtqueue_pop  ◄──────────  1.virtqueue_add(scatterlist)

                   (virtqueue)

                     notify
  3.virtqueue_push ──────────►  4.virtqueue_get_buf()


 The guest then calls the get_buf() function to retrieve completed buffers. To support polling, which is used by network drivers, get_buf() can be called at any time, returning NULL if none have completed. The guest driver can disable further callbacks, at any time, by returning zero from the callback.

            -- https://lwn.net/Articles/239238/
```

- virito笔记补完
    * https://www.cnblogs.com/LoyenWang/p/14589296.html


## TODO: Q

- GFP in kernel means
- 为何virtio使用scatterlist
    * virito数据传递使用scatterlist, 一个scatterlist描述多个buffer
    * 因为直接传递PA, 绕过HPA -> GVA


## ref

- [老版本cuda文档4.0](http://horacio9573.no-ip.org/cuda/index.html)

- ⭐ [VirtIO实现原理——数据传输演示](https://blog.csdn.net/huang987246510/article/details/103708461#_2)
- [virtio简介（五）—— virtio_blk设备分析](https://www.cnblogs.com/edver/p/16255243.html)
- [virtio-net 实现机制【二】（图文并茂）](https://zhuanlan.zhihu.com/p/545258186)
- [深入浅出vhostuser传输模型](https://rexrock.github.io/post/vhu1/)
- [virtIO前后端notify机制详解](https://www.cnblogs.com/ck1020/p/6066007.html)
- [Linux虚拟化KVM-Qemu分析（十一）之virtqueue](https://www.cnblogs.com/LoyenWang/p/14589296.html)
    * TODO: 3.3.1后
- https://royhunter.github.io/2014/08/29/virtio-blk/







