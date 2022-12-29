SRC=$(wildcard *.c)
TARGET=$(patsubst %.cu, % ,$(SRC))

CUDA_H=/usr/local/cuda/include
CUDART_L=/usr/local/cuda/lib64
CUDA_L=/usr/local/cuda/lib64/stubs

all: $(TARGET)
	nvcc -o $(patsubst %.c, %c, $@) -I$(CUDA_H) -L$(CUDART_L) -L$(CUDA_L) $@.c -lcudart -lcuda

clean:
	rm -rf $(TARGET)

.PHONY: all $(TARGET)
