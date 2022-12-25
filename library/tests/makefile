SRC=$(wildcard *.c)
TARGET=$(patsubst %.c, % ,$(SRC))

CUDA_H=/usr/local/cuda/include

all: $(TARGET)

cu: test.cu
	nvcc -cudart=shared test.cu -o test

$(TARGET):
	gcc -o $(patsubst %.c, %c, $@) -I$(CUDA_H) $@.c

clean:
	rm -rf $(TARGET)

.PHONY: all $(TARGET)
