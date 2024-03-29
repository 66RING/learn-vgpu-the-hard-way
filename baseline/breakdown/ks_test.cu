#include<stdio.h>
#include<cuda.h>
#include<chrono>
#include <iostream>

using namespace std;

__global__ void sum(char *x) {
    int tid = threadIdx.x;
    x[tid] = (x[tid] + 1) % 256;
}


int main() {
    //int N = 32;
    //int nbytes = N * sizeof(int);
    char *dx = NULL, *hx = NULL;
    //// 申请显存
    //cudaMalloc((void**)&dx, nbytes);

      int minbyte = 64;
      int maxbyte = 4 << 20;

      for (int i = minbyte; i<=maxbyte; i *= 2) {
        int nbytes = i;
		int loop = 1000;
		double total_time = 0;

		for (int w = 0; w < loop; w++) {

			cudaMalloc((void**)&dx, nbytes);
			hx = (char*)malloc(nbytes);

			for(int j=0;j<nbytes;j++) {
				hx[j] = j % 256;
			}


			cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

			// call GPU
			auto start = chrono::steady_clock::now();
			sum<<<1, nbytes>>>(dx);

			// let gpu finish
			cudaThreadSynchronize();
			total_time += chrono::duration_cast<chrono::microseconds>(end - start).count();

			cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);

			auto end = chrono::steady_clock::now();


			cudaFree(dx);
			free(hx);
		}
        cout << "size(B): " << nbytes << ","
             << total_time / loop
             << ", us" << endl;
      }


    //printf("\n");
    //cudaFree(dx);
    //free(hx);
    return 0;
}



