# CUDA_H=/usr/local/cuda/include

# local develop for now
CUDA_H=/opt/cuda/include

build:
	gcc -c -Wall -Werror -fpic -I$(CUDA_H) libcudart.c
	gcc -shared -Wl,--version-script=shared.version -o libcudart.so.9.0 libcudart.o


clean:
	rm -f *.so *.o *.so.9.0

install:

uninstall:

