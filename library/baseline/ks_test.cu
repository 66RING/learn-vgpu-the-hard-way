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
        auto end = chrono::steady_clock::now();


        cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);


        cout << "size(B): " << nbytes << ","
             << chrono::duration_cast<chrono::microseconds>(end - start).count()
             << " µs" << endl;

        // free(p);
      }


    //printf("\n");
    //cudaFree(dx);
    //free(hx);
    return 0;
}

