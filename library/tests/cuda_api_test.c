#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_profiler_api.h>
#include <stdio.h>

typedef struct cudaDevice {
	CUdevice device;
	CUcontext context;

} cudaDevice;

// TODO: 先一个
cudaDevice devicePool;
int deviceCount = -1;

#define cudaErrorCheck(err) err

static inline void __cudaErrorCheck(cudaError_t err, const int line) {
    char *str;
    if (err != cudaSuccess) {
        str = (char *) cudaGetErrorString(err);
        printf("[CUDA error] %04d \"%s\" line %d\n", err, str, line);
    }
}

static void cudaInit() {
	cudaErrorCheck(cuInit(0));
	cudaErrorCheck(cuDeviceGetCount(&deviceCount));
	printf("cuda device count: %d\n", deviceCount);

	// TODO: 注意create是一个栈, 后进先出
	cudaErrorCheck(cuDeviceGet(&devicePool.device, 0));
	// cudaErrorCheck(cuCtxCreate(&devicePool.context, 0, devicePool.device));
}



const int N = 10;

int cpyTest() {
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

void apiTest() {
	CUmodule m;
	CUfunction f;
	cuModuleLoadData(&m, NULL);
	cuModuleGetFunction(&f, m, "c");
}

int main() {
	cudaInit();
	cpyTest();
	apiTest();
	printf("===cuda test===\n");
}
