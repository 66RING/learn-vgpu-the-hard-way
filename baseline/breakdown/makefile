CU=$(wildcard *.cu)
CU_TARGET=$(patsubst %.cu, % ,$(CU))

CUDA_H=/usr/local/cuda/include
CUDART_L=/usr/local/cuda/lib64
CUDA_L=/usr/local/cuda/lib64/stubs

all: $(CU_TARGET)

$(CU_TARGET):
    nvcc  -std=c++11 -cudart=shared $@.cu -o $(patsubst %.cu, % ,$@)

clean:
    rm -rf $(CU_TARGET)

.PHONY: all $(CU_TARGET)

