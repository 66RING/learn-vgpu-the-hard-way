# 第二阶段开发: API试个遍

## qemu虚拟设备

- handler
    * 从vqueue中获取事件元素抽象`VirtQueueElement* elem`
    * 然后使用`iov_to_buf`从`elem->out_sg`抽象中获取参数, 使用`iov_discard_front`配合删除已用
    * 参数获取完毕, 根据vgpu协议执行对应功能
    * `iov_from_buf`写回iov, TODO: 为何要写回iov
    * `virtqueue_push`, `virtio_notify`写回virtio used队列, 通知驱动
    * 释放资源


## virtio内核驱动

- ref
    * https://www.cnblogs.com/LoyenWang/p/14589296.html
    * https://royhunter.github.io/2014/08/29/virtio-blk/

    * API
        + `iov_to_buf`
            + 直接从iovec中读取数据`iov_to_buf(iov, iovcnt, offset, buf, bytes);
        + `iov_from_buf`
            + 写回`iov`
        + `iov_discard_front`
            + 从vec前端移除bytes个数据, 返回实际移除的数量, 传入iov和iovcnt更新`iov_discard_front(struct iovec **iov, unsigned int *iov_cnt, size_t bytes)`
        + `virtqueue_push`
            + 写回到used队列
        + `virtio_notify`
            + 通知guest驱动
- open
    * 创建fd私有数据(链表), 保存到`filp->private`中
        + TODO: 什么作用呢
    * 向qcuda发送启动命令
    * 仅是`BLOCK_SIZE`的传递
- ioctl
    * 从用户态拷贝数据(`args`)到内核态, `copy_from_user`
    * swtich做响应的功能
    * 数据拷贝回用户态, `copy_to_user`
    * 释放资源
    * API
        + `qcu_cudaMemcpy`
            + `H2D`, `D2H`等用flag标记
            + pA, pB就是`src`, `dst`抽象
- cmd send
    * `sg_init_one`创建两个scatterlist, 分别做收发??
        + `virtqueue_add_sgs` api也要求传入两个list
    * `virtqueue_add_sgs`将scatterlist添加到vring中, 也是在内部转换整buf和desc的添加
        + linux/drivers/virtio/virtio_ring.c
    * `virtqueue_kick`通知另一端
    * `while virtqueue_get_buf()`等待返回
        + `virtqueue_is_broken`而外检测队列没有关闭
    * API
        + `sg_init_one`
            + scatterlist
        + `virtqueue_add_sgs`
            + scatterlist添加到vring中
            + TODO: details
        + `virtqueue_get_buf`
            + get next used buffer, 可以用户检测后端是否处理完成
        + `virtqueue_is_broken`
            + 是否已经关闭
        + `virtqueue_kick`通知另一端
- release
    * 释放open中申请的资源`filp->private`, `kfree`
    * `module_put(THIS_MODULE)`

- `virt_to_phys`


## 用户库





## kernel API

- `kmalloc`
- `INIT_LIST_HEAD(struct list_head)`初始化链表结构
- `scatterlist`
    * 将分散的物理内存以list的性质组织起来
    * page为单位
    * 应用场景:
        + DMA传输时只能以连续物理内存为单位进行, 而大多数情况我们是虚拟内存连续和物理内存不连续, 所以scatterlist就是将不连续的物理内存组织起来
    * `sg_init_one`



## ref

- ⭐ [VirtIO实现原理——数据传输演示](https://blog.csdn.net/huang987246510/article/details/103708461#_2)
- [virtio简介（五）—— virtio_blk设备分析](https://www.cnblogs.com/edver/p/16255243.html)
- [virtio-net 实现机制【二】（图文并茂）](https://zhuanlan.zhihu.com/p/545258186)
- [深入浅出vhostuser传输模型](https://rexrock.github.io/post/vhu1/)
- [virtIO前后端notify机制详解](https://www.cnblogs.com/ck1020/p/6066007.html)







