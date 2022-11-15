ifneq ($(KERNELRELEASE),)
	obj-m := vgpu_driver.o
else
	# KERN_DIR:=/lib/modules/6.0.7-artix1-1/build
	KERN_DIR:=/lib/modules/$(shell uname -r)/build
	PWD := $(shell pwd)

build:
	$(MAKE) -C $(KERN_DIR) M=$(PWD) modules
endif

install: build
	@if [ -n "`lsmod | grep vgpu_driver`" ]; then \
	sudo rmmod vgpu_driver; \
	fi
	sudo dmesg -C
	sudo insmod vgpu_driver.ko
	sudo dmesg

clean:
	@if [ -n "`lsmod | grep vgpu_driver`" ]; then \
	sudo rmmod vgpu_driver; \
	fi
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions *.mod *.symvers *.order *.ur-safe

