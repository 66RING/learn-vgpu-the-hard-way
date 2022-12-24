#include "../libcudart.c"

const int N = 10;

int main() {
	int *dx = NULL;
	int *hx = NULL;
	int *nhx = NULL;
	int nbyte = N * sizeof(int);
	printf("origin: dx %p, hx %p\n", dx, hx);

	cudaMalloc((void**)&dx, nbyte);

	hx = (int*)malloc(nbyte);
	nhx = (int*)malloc(nbyte);

	for (int i=0;i<N;i++) {
		hx[i] = i;
	}

	printf("dx %p, hx %p, nhx %p origin is:\n", dx, hx, nhx);
	printf("hx: ");
	for (int i=0;i<N;i++) {
		printf("%d ", hx[i]);
	}
	printf("\n");

	printf("nhx: ");
	for (int i=0;i<N;i++) {
		printf("%d ", nhx[i]);
	}
	printf("\n");

	printf("hx in bytes: ");
	for (int i=0;i < nbyte; i++) {
		printf("0x%x ", *((uint8_t *)hx+i));
	}
	printf("\n");

	// cudaErrorCheck(err = cuMemcpyHtoD(dx, hx, sizeof(int) * N));
	cudaMemcpy(dx, hx, sizeof(int) * N, cudaMemcpyHostToDevice);

	for (int i=0;i<N;i++) {
		hx[i] = 0xff;
	}


	// cudaErrorCheck(err = cuMemcpyDtoH(nhx, dx, sizeof(int) * N));
	cudaMemcpy(nhx, dx, sizeof(int) * N, cudaMemcpyDeviceToHost);

	printf("\nafter D2H\n");

	printf("dx %p, hx %p, nhx %p after is:\n", dx, hx, nhx);
	printf("hx: ");
	for (int i=0;i<N;i++) {
		printf("%d ", hx[i]);
	}
	printf("\n");

	printf("nhx: ");
	for (int i=0;i<N;i++) {
		printf("%d ", nhx[i]);
	}
	printf("\n");

	cudaFree(dx);
	free(hx);
}
