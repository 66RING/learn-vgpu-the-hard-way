# 第一阶段开发: 打印机

不实现任何逻辑, 只熟悉框架, 流畅触发回调函数打印信息。


## 最小qemu虚拟设备

## 最小virtio设备驱动

1. 注册驱动`probe`和`remove`方法
    - 填写`struct virtio_driver`结构体填表, 然后在`init()`中使用`register_virtio_driver()`注册驱动
2. `probe()`中注册设备文件操作方法, 创建设备文件
    - 驱动注册后会使用`probe()`探测设备, 这时要求注册时的`id_table`与底层设备匹配
    - `probe()`中负责创建设备文件, 之类使用`misc_register`注册并自动生成设备文件
3. 设备使用
    - `probe()`结束后设备文件创建完成 
    - 对设备文件操作会触发绑定的`fops`中的`open`, `ioctl`, `release`

- ref
    * linux/drivers/nvdimm/virtio_pmem.c
    * https://paper.seebug.org/779/
    * probe何时调用
        + https://www.cnblogs.com/hoys/archive/2011/04/01/2002299.html
        + 在总线上驱动和设备的名字匹配，就会调用驱动的probe函数
        + 设备id和驱动注册的id: `id_table`中id
        + 当id被一个驱动占用时就不能被另一个驱动使用


### virtio API

- `struct virtio_device.priv`
    * probe时初始化, 保存自定义的virtio设备的结构体
- `struct virtqueue`
    * virtio处理vring的接口
- `virtio_find_single_vq()`
    * 为virtio设备生成一个`virtqueue`
    * TODO: 更多细节

- ref
    * https://blog.csdn.net/xidianjiapei001/article/details/89299775
    * https://www.cnblogs.com/edver/p/16255243.html


### API

- `register_virtio_driver()`
    * virtio驱动注册接口, `init`时调用, 填表注册
    * `unregister_virtio_driver()`释放, `exit`
- `misc_register()`
    * 在系统创建特殊设备文件, 填表注册
        + `struct file_operations {}`
        + `struct miscdevice {}`
    * `probe`时刻创建
    * 注册杂项设备
    * 主设备号为10的设备, 次设备号由`.minor`设置
    * 注册时自动创建字符设备
    * `misc_deregister()`释放, `remove`

- 内核态内存申请
    * `kmalloc()`
        + 线性申请内存, 物理空间连续, `brk()`
        + 不能超过128KB
        + `kfree()`释放
    * `kzalloc()`
        + `kmalloc`后还对申请的内存进行清零
        + `kfree()`释放
    * `vmalloc`
        + 虚拟地址连续, 但物理地址不一定连续
        + 对申请大小没有限制
        + `vfree()`释放


### ref

- https://zhuanlan.zhihu.com/p/144349599

## 最小用户态测试程序
