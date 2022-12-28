CUDA_H=/usr/local/cuda/include

build:
	gcc -c -Wall -Werror -fpic -I$(CUDA_H) libcudart.c
	gcc -shared -Wl,--version-script=shared.version -o libcudart.so.9.0 libcudart.o


clean:
	rm -f *.so *.o *.so.9.0

install:
	sudo cp libcudart.so.9.0 /usr/local/lib/libcudart.so.9.0
	sudo ln -sf libcudart.so.9.0 /usr/local/lib/libcudart.so
	sudo ldconfig


uninstall:
	sudo rm /usr/local/lib/libcudart.so.9.0
	sudo rm /usr/local/lib/libcudart.so
	sudo ldconfig

.PHONY: build

