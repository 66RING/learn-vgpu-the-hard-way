SRC=$(wildcard *.cu)
TARGET=$(patsubst %.cu, % ,$(SRC))

CUDA_H=/usr/local/cuda/include
CUDART_L=/usr/local/cuda/lib64
CUDA_L=/usr/local/cuda/lib64/stubs

build: $(TARGET)

$(TARGET):
	nvcc -o $(patsubst %.cu, %, $@) -I$(CUDA_H) -L$(CUDART_L) -L$(CUDA_L) $@.cu -lcudart -lcuda

lib:
	gcc -c -Wall -Werror -fpic -I$(CUDA_H) libcudart.c
	gcc -shared -Wl,--version-script=shared.version -o libcudart.so.9.0 libcudart.o

clean:
	rm -rf $(TARGET) *.o

.PHONY: all $(TARGET) lib build
