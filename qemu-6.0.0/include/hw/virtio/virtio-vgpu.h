/*
 * Virtio Vgpu device
 *
 * Copyright (C) 2018-2019 66RING
 *
 * This work is licensed under the terms of the GNU GPL, version 2.
 * See the COPYING file in the top-level directory.
 */

#ifndef HW_VIRTIO_VGPU_H
#define HW_VIRTIO_VGPU_H

#include "hw/virtio/virtio.h"
#include "qapi/qapi-types-machine.h"
#include "qom/object.h"

#define TYPE_VIRTIO_VGPU "virtio-vgpu"

OBJECT_DECLARE_SIMPLE_TYPE(VirtIOVGPU, VIRTIO_VGPU)

#define VIRTIO_VGPU_ADDR_PROP "memaddr"
#define VIRTIO_VGPU_MEMDEV_PROP "memdev"

#define assertm(exp, msg) assert(((void)msg, exp))

#define panic(msg) fprintf(stderr, msg);abort()

typedef struct VirtIOVgpuConf VirtIOVgpuConf;

struct VirtIOVgpuConf {
	uint64_t mem_size;
};

struct VirtIOVGPU {
    VirtIODevice parent_obj;
	VirtIOVgpuConf conf;
    VirtQueue *vq;
};

#endif

