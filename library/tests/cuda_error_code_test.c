#include "../libcudart.c"
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_profiler_api.h>


const int N = 10;

int main() {
	int *ptr;
	// cudaMalloc((void**)&ptr, sizeof(int) * N);
	printf("%p\n", ptr);
	int err = cudaFree((void*)ptr);
	printf("expect 35: get %d", err);

	return 0;
}

