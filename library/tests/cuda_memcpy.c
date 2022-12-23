#include "../libcudart.c"

const int N = 10;

int main() {
	int *dx = NULL;
	int *hx = NULL;

	cudaMalloc((void**)&dx, sizeof(int) * N);
	hx = (int*)malloc(N);
	for (int i=0;i<N;i++) {
		hx[i] = i;
	}

	printf("dx %p, hx %p\n", dx, hx);
	for (int i=0;i < sizeof(int) * N; i++) {
		printf("0x%x ", *((uint8_t *)hx+i));
	}
	printf("\n");

	cudaMemcpy(dx, hx, sizeof(int) * N, cudaMemcpyHostToDevice);

	cudaFree(dx);
	free(hx);
}
