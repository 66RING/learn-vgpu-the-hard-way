- `make -n`
- `ldd`
- `nm`

编译时引入cuda库

```
./configure --target-list="x86_64-softmmu x86_64-linux-user" --enable-debug-info \
    --extra-ldflags=-L/home/ub/gpgpu-sim_distribution/lib/gcc-5.5.0/cuda-9000/release \
    --extra-cflags=-lcuda \
    --extra-cflags=-lcudart \
```
