- `make -n`
- `ldd`
- `nm`

编译时引入cuda库

```
    --extra-ldflags=-L/home/ub/gpgpu-sim_distribution/lib/gcc-5.5.0/cuda-9000/release \
./configure --target-list="x86_64-softmmu x86_64-linux-user" --enable-debug-info \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --extra-ldflags=-L/usr/local/cuda/lib64/stubs \
    --extra-cflags=-lcuda \
    --extra-cflags=-lcudart \
```

- 命令行启动qemu除了需要`-nographic`还要在内核启动参数中加`console=ttyS0`
- 驱动安装多种参数`sudo sh ./cuda_9.2.148_396.37_linux.run --silent --verbose --driver --toolkit --samples`
