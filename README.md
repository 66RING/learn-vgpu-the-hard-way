# gpu虚拟化研究

> 基于API remoting

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
