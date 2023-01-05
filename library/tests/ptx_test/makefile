SRC=$(wildcard *.cu)
TARGET=$(patsubst %.cu, % ,$(SRC))

CUDA_H=/usr/local/cuda/include
CUDART_L=/usr/local/cuda/lib64
CUDA_L=/usr/local/cuda/lib64/stubs

$(TARGET):
	nvcc -o $(patsubst %.cu, %, $@) -I$(CUDA_H) -L$(CUDART_L) -L$(CUDA_L) $@.cu -lcudart -lcuda

clean:
	rm -rf $(TARGET)

.PHONY: all $(TARGET)
