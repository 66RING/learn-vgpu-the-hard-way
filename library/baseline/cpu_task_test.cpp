#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using namespace std;

int main() {
  char *dx = NULL, *hx = NULL;

  int minbyte = 64;
  int maxbyte = 4 << 20;

  for (int i = minbyte; i <= maxbyte; i *= 2) {
    int nbytes = i;

    auto start = chrono::steady_clock::now();
    dx = (char *)malloc(nbytes);
    hx = (char *)malloc(nbytes);

    for (int j = 0; j < nbytes; j++) {
      hx[j] = j % 256;
    }

    memcpy(dx, hx, nbytes);
    // cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

    // call GPU
    // sum<<<1, nbytes>>>(dx);
    for (int i = 0; i < nbytes; i++) {
      dx[i] = (dx[i] + i) % 256;
    }

    // let gpu finish
    // cudaThreadSynchronize();

    // cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
    memcpy(hx, dx, nbytes);

    auto end = chrono::steady_clock::now();

    cout << "size(B): " << nbytes << ","
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " µs" << endl;
  }
  return 0;
}
