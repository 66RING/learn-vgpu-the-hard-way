#include "../libcudart.c"

const int N = 10;

int main() {
	int *ptr;
	cudaMalloc((void**)&ptr, sizeof(int) * N);
	printf("%p\n", ptr);
	cudaFree((void*)ptr);

	return 0;
}
