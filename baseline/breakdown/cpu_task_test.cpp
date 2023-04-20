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
	int loop = 1000;
	double total_time = 0;

	for (int w = 0; w < loop; w++) {
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

	  total_time += chrono::duration_cast<chrono::microseconds>(end - start).count();
	  free(dx);
	  free(hx);
	}

    cout << "size(B): " << nbytes << ","
         << total_time / loop
         << " µs" << endl;
  }
  return 0;
}
