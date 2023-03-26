#include<stdio.h>
#include<cuda.h>
#include<chrono>
#include <iostream>

using namespace std;

int main() {
    //int N = 32;
    //int nbytes = N * sizeof(int);
    int *dx = NULL, *hx = NULL;
    //// 申请显存
    //cudaMalloc((void**)&dx, nbytes);

      int minbyte = 64;
      int maxbyte = 4 << 20;

      for (int i = minbyte; i<=maxbyte; i *= 2) {

        int nbytes = i;
        auto start = chrono::steady_clock::now();
        cudaMalloc((void**)&dx, nbytes);
        auto end = chrono::steady_clock::now();

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

