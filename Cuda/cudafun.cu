#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include "readjpeg.h"

#define KERNX 5 //this is the x-size of the kernel. It will always be odd.
#define KERNY 5 //this is the y-size of the kernel. It will always be odd.

double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

__global__ void convolve(
        float * pad, 
        float * kern, 
        float * out, 
        const int pad_num_col,
        const float median_index) 
{ 
        const int KER_SIZE = 5;
        const int NUM_ITERATIONS = 8;

        const int out_num_col = gridDim.x*blockDim.x;
        const int out_col = blockIdx.x*blockDim.x+threadIdx.x; 
        const int out_row = blockIdx.y*blockDim.y+threadIdx.y;
        float buffer[KER_SIZE*KER_SIZE];

        int buffer_row_head;
        int pad_row_head;
        int i = 0;

        // copy into buffer
        for (int row = 0; row < KER_SIZE; row++) {
                buffer_row_head = row * KER_SIZE;
                pad_row_head = (row+out_row) * pad_num_col + out_col;

                for (int col = 0; col < KER_SIZE; col++) { 
                        i = buffer_row_head+col;
                        buffer[i] = pad[pad_row_head+col];
                }
        }

        // find median with t dim3 blockSize(1, 1, 1);
        float estimate = 128.0f;
        float lower = 0.0f;
        float upper = 255.0f;
        float higher;

        for (int _ = 0; _ < NUM_ITERATIONS; _++){
                higher = 0;
                for (int i = 0; i < KER_SIZE*KER_SIZE; i++){
                        higher += ((float)(estimate < buffer[i])) * kern[i];
                }
                if (higher > median_index){
                        lower = estimate;
                } else {
                        upper = estimate;
                }
                estimate = 0.5 * (upper + lower);
        }


        out[out_row*out_num_col+out_col] = estimate;

} 


void cuda_function(int data_size_X, int data_size_Y, float* kernel, float* in, float* out, double* t0, double* t1) {
	printf("Initiating CUDA...\n");
    

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
    cudaMalloc((void**) &g_in, sizeof(float)*pad_size_total);
    cudaMalloc((void**) &g_out, sizeof(float)*data_size_X*data_size_Y);
    cudaMalloc((void**) &g_kern, sizeof(float)*KERNY*KERNX);

    cudaMemcpy(g_in, in_cpy, sizeof(float)*pad_size_total, cudaMemcpyHostToDevice);
    cudaMemcpy(g_kern, kern_cpy, sizeof(float)*KERNX*KERNY, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

	float sum = 0;
	for (int i=0; i<KERNX*KERNY; i++){
	        sum += kernel[i];
	}
	sum /= 2;

 	const dim3 blockSize(1, 1, 1);
  	const dim3 gridSize(data_size_X, data_size_Y, 1);

	// launch the kernel
 	*t0 = timestamp();
  	convolve<<<gridSize, blockSize>>>(g_in, g_kern, g_out, pad_size_X, sum);
   	*t1 = timestamp();
  	cudaDeviceSynchronize();

  	// copy back the result array to the CPU
  	cudaMemcpy(out, g_out, sizeof(int)*data_size_X*data_size_Y, cudaMemcpyDeviceToHost);
    	cudaDeviceSynchronize();

  	cudaFree(g_in);
  	cudaFree(g_out);
	cudaFree(g_kern);

  	return;
}
