# gpu虚拟化研究

> 基于API remoting

## outline

- [stage1](./doc/dev_step1.md)
    * 简单的内核驱动探测: probe到就打印点什么
    * qemu虚拟设备创建: 内核能在`/dev/xxx`识别到virtio设备
- [stage2](./doc/dev_step2.md)
    * 完善内核驱动: 简单的ioctl用户态内核态通信, echo一下就行
    * 简单的host, guest通信, 简单输出点什么, 如一个int `0xdeadbeef`
    * 引入cuda driver API, 使qemu能够编译成功
    * 分析最小cuda程序所需API
    * 学习API: 内核驱动的API, VMM的API
- [stage3](./doc/dev_step3.md)
    * 完善程序: 使guest能跑通最小的cuda程序
    * 逆向分析各个cuda driver API的用法
    * 理清cuda程序本身的执行流程


## idea tips

- vhost
    * vhost-user back ends are way to service the request of VirtIO devices outside of QEMU itself.
    * https://qemu.readthedocs.io/en/latest/system/devices/vhost-user.html
- gpu scheduler: 仅实现了任务接收和FIFO派发, 没有"中断再恢复"
    * https://github.com/ExpectationMax/simple_gpu_scheduler
    * 使用`tail -f -n 0 queue.file | cmd`做命令接收队列
    * 信号量pv操作做gpu资源限制, 这样可以使用它内置的等待队列

## refs

- [qcuda](https://github.com/coldfunction/qCUDA)
