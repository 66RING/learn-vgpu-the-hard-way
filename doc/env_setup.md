# 环境搭建

- Ubuntu18.04

依赖安装

```sh
# 如果qemu在ub18编译
sudo apt install -y  pkg-config bridge-utils uml-utilities zlib1g-dev libglib2.0-dev autoconf \
automake libtool libsdl1.2-dev libsasl2-dev libcurl4-openssl-dev libsasl2-dev libaio-dev libvde-dev \
libsdl2-dev libaio-dev  libattr1-dev libbrlapi-dev libcap-ng-dev libgnutls28-dev libgtk-3-dev libiscsi-dev liblttng-ust-dev \
libncurses5-dev libnfs-dev libnss3-dev \
libpixman-1-0 libpng-dev librados-dev libsdl1.2-dev libseccomp-dev libcapstone3 \
libspice-protocol-dev libspice-server-dev libssh2-1-dev liburcu-dev libusb-1.0-0-dev libvte-dev sparse uuid-dev \
libspice-server1 libbrlapi0.6 libnettle6 libiscsi7 \
```

## 虚拟机网络

使用`dhclient -v`分配ip后即可接入网络。


## 驱动安装后

设备文件`/dev/xxx`将会被创建, 通过对设备文件的操作与驱动交互


## Option: GPGPU-sim
