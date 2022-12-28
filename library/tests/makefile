SRC=$(wildcard *.c)
TARGET=$(patsubst %.c, % ,$(SRC))

CUDA_H=/usr/local/cuda/include
CUDART_L=/usr/local/cuda/lib64
CUDA_L=/usr/local/cuda/lib64/stubs

all: $(TARGET)

cu: test.cu
	nvcc -cudart=shared test.cu -o test

$(TARGET):
	gcc -o $(patsubst %.c, %c, $@) -I$(CUDA_H) -L$(CUDART_L) -L$(CUDA_L) $@.c -lcudart -lcuda

clean:
	rm -rf $(TARGET)

.PHONY: all $(TARGET)
