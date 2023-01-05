SRC=$(wildcard *.c)
TARGET=$(patsubst %.c, % ,$(SRC))

CUDA_H=/usr/local/cuda/include
CUDART_L=/usr/local/cuda/lib64
CUDA_L=/usr/local/cuda/lib64/stubs

all: $(TARGET)

cu: test.cu
	# nvcc -I$(CUDA_H) -L$(CUDART_L) -L$(CUDA_L) -cudart=shared test.cu -o test -lcuda -lcudart
	# shared导致image无法识别"CUDA_ERROR_UNKNOWN"

	# 静态导致ptx不合法"CUDA_ERROR_INVALID_PTX"
	nvcc -I$(CUDA_H) -L$(CUDART_L) -L$(CUDA_L) test.cu -o test -lcuda -lcudart

$(TARGET):
	gcc -o $(patsubst %.c, %c, $@) -I$(CUDA_H) -L$(CUDART_L) -L$(CUDA_L) $@.c -lcudart -lcuda

clean:
	rm -rf $(TARGET)

.PHONY: all $(TARGET)
