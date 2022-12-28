## 最小化实现

### 设备本体

### pci设备

```
/**
 * DECLARE_INSTANCE_CHECKER:
 * @InstanceType: instance struct name
 * @OBJ_NAME: the object name in uppercase with underscore separators
 * @TYPENAME: type name
 *
 * Direct usage of this macro should be avoided, and the complete
 * OBJECT_DECLARE_TYPE macro is recommended instead.
 *
 * This macro will provide the instance type cast functions for a
 * QOM type.
 */
 ```

TODO: more note maybe


## 功能完善

- 使用driver API需要先初始化设备
- 使用gpgpu-sim需要将config文件拷贝到当前目录
    * 不能用root, 否则工作目录和环境都会改变
- gpgpu-sim上有点函数没有实现, 所以需要先场外测试一下:
    * cuMemcpyDtoH
    * cuMemcpyHtoD
    * cuCtxCreate
    * 因此部分功能暂且搁置


### 面向gpgpu-sim编程(放弃)

- 找fatBin中介文件
    * 这里fatCubin是用不上的, 它可以直接通过二进制文件找到, 然后解析, 记录到gpgpu-sim的一个表中, 然后返回handle给RegiterFunction
    * 验证: gdb断点`__cudaRegisterFatBinary`, 然后gdb修改`fatCubin`参数的值`set fatCubin=0`, 执行依旧正常
    * ⭐ 但是gpgpu-sim通过`get_app_binary()`获取二进制文件, vmm环境下外面的gpgpu-sim怎么拿到vmm内部的用户态的二进制文件呢？


- ❌ 所以这里只需要向back发送一个`__cudaRegisterFatBinary`请求, 然后包backend返回的handle返回即可
- 所以考虑一下直接`__cudaRegisterFunction`
- 而且还要看得更长远些, 要看到`cudaLaunch`

总之很麻烦还不通用不现实, 遂放弃

```
// The extracted file name is associated with a fat_cubin_handle passed
// into cudaLaunch().  Inside cudaLaunch(), the associated file name is
// used to find the PTX/SASS section from cuobjdump, which contains the
// PTX/SASS code for the launched kernel function.
// This allows us to work around the fact that cuobjdump only outputs the
// file name associated with each section.
```


### 带gpu的云服务器

- 你的gpu呢?
    * 轻薄本没有, 实验室发的我还不好意思找老板拿(没干活)

所以我只能编译好后传过去




## misc

### 为什么重启后make失败了?

gcc版本问题: 因为我在ubuntu20安装了低版本的gcc, 它原本是gcc-7

修改系统gcc版本, 添加到可用项, 最后的100表示优先级(越大优先级越高)适用于自动模式

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 100

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100

# 删除用
sudo update-alternatives --remove /usr/bin/gcc gcc /usr/bin/gcc-5 100
sudo update-alternatives --remove /usr/bin/g++ g++ /usr/bin/g++-5 100
```

之后手动修改版本

```
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

