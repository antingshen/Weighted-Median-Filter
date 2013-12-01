#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include "readjpeg.h"
#include "clhelp.h"

#define KERNX 5 //this is the x-size of the kernel. It will always be odd.
#define KERNY 5 //this is the y-size of the kernel. It will always be odd.
void cuda_function(int data_size_X, int data_size_Y, float* kernel, float* in, float* out, double* t0, double* t1) {
	printf("Initiating OpenCL...\n");
    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;
    int pad_size_X = data_size_X+2*kern_cent_X;
    int pad_size_Y = data_size_Y+2*kern_cent_Y;
    int pad_size_total = pad_size_Y * pad_size_X;

    // Padding code
    float kern_cpy[KERNX*KERNY];
    float *in_cpy = (float*) calloc(pad_size_total, sizeof(float));

    for (int i = 0; i<KERNX*KERNY; i++) {
        kern_cpy[i] = kernel[KERNX * KERNY - 1 - i];
    }
    for (int b = 0; b < data_size_Y; b++){
        for (int c = 0; c < data_size_X; c++){
            in_cpy[(pad_size_X+kern_cent_X)+(c+b*data_size_X)+(b*2*kern_cent_X)] = in[c+b*data_size_X];
        }
    }

    float *g_in;
    float *g_out;
    float *g_kern;
    cudaMalloc((void**) &g_in, sizeof(int)*pad_size_total);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    cudaMalloc((void**) &g_out, sizeof(int)*data_size_X*data_size_Y);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    cudaMalloc((void**) &g_kern, sizeof(int)*KERNY*KERNX);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    cudaMemcpy(g_in, in_cpy, sizeof(int)*pad_size_total, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    cudaMemcpy(g_kern, kern_cpy, sizeof(int)*KERNX*KERNY, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
	float sum = 0;
	for (int i=0; i<KERNX*KERNY; i++){
	        sum += kernel[i];
	}
	sum /= 2;
   	dim3 block_size;
  	block_size.x = data_size_X;
  	block_size.y = data_size_Y;
  	// launch the kernel
    *t0 = timestamp();
  	convolve<<<1, block_size>>>(g_in, g_kern, g_out, pad_size_X, sum);
    *t1 = timestamp();
  	cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

  	// copy back the result array to the CPU
  	cudaMemcpy(out, g_out, sizeof(int)*data_size_X*data_size_Y, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

  	cudaFree(g_in);
  	cudaFree(g_out);
	cudaFree(g_kern);
  	return 1;
}