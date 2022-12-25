#include<stdio.h>
#include<cuda.h>

typedef double FLOAT;
__global__ void sum(FLOAT *x) {
	int tid = threadIdx.x;
	x[tid] += 1;
}

int main() {
	int N = 32;
	int nbytes = N * sizeof(FLOAT);

	FLOAT *dx = NULL, *hx = NULL;
	int i;
	// 申请显存
	cudaMalloc((void**)&dx, nbytes);
	
	// 申请成功
	if (dx == NULL) {
		printf("GPU alloc fail");
		return -1;
	}

	// 申请CPU内存
	hx = (FLOAT*)malloc(nbytes);
	if (hx == NULL) {
		printf("CPU alloc fail");
		return -1;
	}

	// init: hx: 0..31
	printf("hx original:\n");
	for(int i=0;i<N;i++) {
		hx[i] = i;
		printf("%g\n", hx[i]);
	}

	// copy to GPU
	cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

	// call GPU
	sum<<<1, N>>>(dx);

	// let gpu finish
	cudaThreadSynchronize();

	// copy data to CPU
	cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);

	printf("hx after:\n");
	for(int i=0;i<N;i++) {
		printf("%g\n", hx[i]);
	}
	cudaFree(dx);
	free(hx);
	return 0;
}

