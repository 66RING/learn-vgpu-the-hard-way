#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_profiler_api.h>
#include <stdio.h>

#define checkErrors(err) __cudaErrorCheck(err, __LINE__)
static inline void __cudaErrorCheck(CUresult err, const int line) {
    char *str;
    if (err != cudaSuccess) {
		cuGetErrorName(err, (const char **) &str);
        printf("[CUDA error] %04d \"%s\" line %d\n", err, str, line);
    }
}

#define MAX_CTX_CNT 1

CUdevice device;
size_t totalGlobalMem;
CUcontext _context[MAX_CTX_CNT];



CUresult initCUDA()
{
    int deviceCount = 0;
    // 初始化cuda driver API，这个初始化必须在使用任何driver api前完成
    // 这一步经常会由于你的cuda driver和kernel driver不匹配而失败
    CUresult err = cuInit(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
        checkErrors(cuDeviceGetCount(&deviceCount));

    // 获取第一个CUDA设备
    checkErrors(cuDeviceGet(&device, 0));

    // 获取compute capabilities和device名称，
    checkErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    // 获取device的video memory容量
    checkErrors( cuDeviceTotalMem(&totalGlobalMem, device) );
    printf("  Total amount of global memory:   %llu bytes\n",
           (unsigned long long)totalGlobalMem);
    printf("  64-bit Memory Address:           %s\n",
           (totalGlobalMem > (unsigned long long)4*1024*1024*1024L)?
           "YES" : "NO");

    // 可以创建一个或多个CUDA context
    for (int ctxCnt = 0; ctxCnt < MAX_CTX_CNT; ctxCnt++)
    {
        err = cuCtxCreate(&_context[ctxCnt], 0, device);
        if (err != CUDA_SUCCESS) {
            printf("* Error initializing the CUDA context, error code is %d.\n", err);
            return err;
        }
    }
	return err;
}

CUfunction function;
char* module_file = ".ptx";
char* kernel_name = "sum";
CUdeviceptr d_a, d_b, d_c;
int block_size = 32;

CUresult loadKernelFunction()
{
	CUmodule module;
	CUresult err;
    // 这里的module_file是nvcc将kernel code编译成的ptx文件，这里用的是offline static compilation。
    // 也可以使用nvrtc实现online comilation。产生后的PTX代码，使用cuModuleLoadData加载module，使用cuLinkAddData进行link。
    // 也可以通过cuModuleLoadFatBinary直接导入fatbin文件 
    err = cuModuleLoad(&module, module_file);

    err = cuModuleGetFunction(&function, module, kernel_name);

    return err;
}

void release()
{
    checkErrors( cuMemFree(d_a) );
    checkErrors( cuMemFree(d_b) );
    checkErrors( cuMemFree(d_c) );

    for (int ctxCnt = 0; ctxCnt < MAX_CTX_CNT; ctxCnt++)
        cuCtxDestroy(_context[ctxCnt]);
}
 
int main()
{
    int n = 10;
    int a[n], b[n], c[n];

    // 在host端初始化array
    for (int i = 0; i < n; ++i) {
        a[i] = n - i;
        b[i] = i * i;
    }

    initCUDA();
    loadKernelFunction();

    // 在device端分配内存
    checkErrors( cuMemAlloc(&d_a, sizeof(int) * n) );
    checkErrors( cuMemAlloc(&d_b, sizeof(int) * n) );
    checkErrors( cuMemAlloc(&d_c, sizeof(int) * n) );

    // 将array从host拷贝到device
    checkErrors( cuMemcpyHtoD(d_a, a, sizeof(int) * n) );
    checkErrors( cuMemcpyHtoD(d_b, b, sizeof(int) * n) );

    checkErrors( cuLaunchKernel(function, 
                                    (n+block_size-1)/block_size, 1, 1,  // Grid dim
                                    block_size, 1, 1,                   // Threads dim
                                    0, 0, args, 0) );

    // checkErrors( loadKernelFunction());

    // 需要将结果从device再拷贝回host，省略、、、
    release();
}


