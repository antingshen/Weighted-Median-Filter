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
typedef struct
{
        float r;
        float g;
        float b;
} pixel_t;
__global__ void convolveBW(
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
__global__ void convolveC(
        pixel_t * pad, 
        float * kern, 
        pixel_t * out, 
        const int pad_num_col,
        const float median_index
        ) 
{ 
        const int NUM_ITERATIONS = 8;

        const int out_num_col = gridDim.x*blockDim.x;
        const int out_col = blockIdx.x*blockDim.x+threadIdx.x; 
        const int out_row = blockIdx.y*blockDim.y+threadIdx.y;

        float buffer[25];
        pixel_t pix;

        int pad_row_head;

        // copy into buffer
        pad_row_head = out_row * pad_num_col + out_col;
        pix = pad[pad_row_head];   buffer[0] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+1]; buffer[1] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+2]; buffer[2] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+3]; buffer[3] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+4]; buffer[4] = pix.r + pix.g + pix.b;

        pad_row_head += pad_num_col;
        pix = pad[pad_row_head];   buffer[5] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+1]; buffer[6] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+2]; buffer[7] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+3]; buffer[8] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+4]; buffer[9] = pix.r + pix.g + pix.b;

        pad_row_head += pad_num_col;
        pix = pad[pad_row_head];   buffer[10] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+1]; buffer[11] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+2]; buffer[12] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+3]; buffer[13] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+4]; buffer[14] = pix.r + pix.g + pix.b;

        pad_row_head += pad_num_col;
        pix = pad[pad_row_head];   buffer[15] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+1]; buffer[16] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+2]; buffer[17] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+3]; buffer[18] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+4]; buffer[19] = pix.r + pix.g + pix.b;

        pad_row_head += pad_num_col;
        pix = pad[pad_row_head];   buffer[20] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+1]; buffer[21] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+2]; buffer[22] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+3]; buffer[23] = pix.r + pix.g + pix.b;
        pix = pad[pad_row_head+4]; buffer[24] = pix.r + pix.g + pix.b;

        // find median with binary search
        float estimate = 382.5f;
        float lower = 0.0f;
        float upper = 765.0f;
        float higher;

        for (int _ = 0; _ < NUM_ITERATIONS; _++){
                higher = 0;
                higher += ((float)(estimate < buffer[0])) * kern[0];
                higher += ((float)(estimate < buffer[1])) * kern[1];
                higher += ((float)(estimate < buffer[2])) * kern[2];
                higher += ((float)(estimate < buffer[3])) * kern[3];
                higher += ((float)(estimate < buffer[4])) * kern[4];
                higher += ((float)(estimate < buffer[5])) * kern[5];
                higher += ((float)(estimate < buffer[6])) * kern[6];
                higher += ((float)(estimate < buffer[7])) * kern[7];
                higher += ((float)(estimate < buffer[8])) * kern[8];
                higher += ((float)(estimate < buffer[9])) * kern[9];
                higher += ((float)(estimate < buffer[10])) * kern[10];
                higher += ((float)(estimate < buffer[11])) * kern[11];
                higher += ((float)(estimate < buffer[12])) * kern[12];
                higher += ((float)(estimate < buffer[13])) * kern[13];
                higher += ((float)(estimate < buffer[14])) * kern[14];
                higher += ((float)(estimate < buffer[15])) * kern[15];
                higher += ((float)(estimate < buffer[16])) * kern[16];
                higher += ((float)(estimate < buffer[17])) * kern[17];
                higher += ((float)(estimate < buffer[18])) * kern[18];
                higher += ((float)(estimate < buffer[19])) * kern[19];
                higher += ((float)(estimate < buffer[20])) * kern[20];
                higher += ((float)(estimate < buffer[21])) * kern[21];
                higher += ((float)(estimate < buffer[22])) * kern[22];
                higher += ((float)(estimate < buffer[23])) * kern[23];
                higher += ((float)(estimate < buffer[24])) * kern[24];
                if (higher > median_index){
                        lower = estimate;
                } else {
                        upper = estimate;
                }
                estimate = 0.5 * (upper + lower);
        }

        float diff;
        for (int i=0; i<25; i++){
                diff = (buffer[i] - estimate)/3;
                diff = diff * diff;
                if (diff <= 1){
                        out[out_row*out_num_col+out_col] = pad[(i/5+out_row)*pad_num_col+i%5+out_col];
                        return;
                }
        }

        pix.r = 255; pix.g = 0; pix.b = 0;
        out[out_row*out_num_col+out_col] = pix;
        return;
} 



void cuda_function(int data_size_X, int data_size_Y, float* kernel, pixel_t* in, pixel_t* out, double* t0, double* t1) {
	printf("Initiating CUDA Color...\n");
    

    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;
    int pad_size_X = data_size_X+2*kern_cent_X;
    int pad_size_Y = data_size_Y+2*kern_cent_Y;
    int pad_size_total = pad_size_Y * pad_size_X;

    // Padding code
    float kern_cpy[KERNX*KERNY];

    for (int i = 0; i<KERNX*KERNY; i++) {
        kern_cpy[i] = kernel[KERNX * KERNY - 1 - i];
    }
            pixel_t *in_cpy = (pixel_t*) calloc(pad_size_total, sizeof(pixel_t));
    for (int b = 0; b < data_size_Y; b++){
        for (int c = 0; c < data_size_X; c++){
            in_cpy[(pad_size_X+kern_cent_X)+(c+b*data_size_X)+(b*2*kern_cent_X)].r = in[c+b*data_size_X].r;
            in_cpy[(pad_size_X+kern_cent_X)+(c+b*data_size_X)+(b*2*kern_cent_X)].g = in[c+b*data_size_X].g;
            in_cpy[(pad_size_X+kern_cent_X)+(c+b*data_size_X)+(b*2*kern_cent_X)].b = in[c+b*data_size_X].b;
        }
    }

    pixel_t *g_in;
    pixel_t *g_out;
    float *g_kern;
    cudaMalloc((void**) &g_in, sizeof(pixel_t)*pad_size_total);
    cudaMalloc((void**) &g_out, sizeof(pixel_t)*data_size_X*data_size_Y);
    cudaMalloc((void**) &g_kern, sizeof(float)*KERNY*KERNX);

    cudaMemcpy(g_in, in_cpy, sizeof(pixel_t)*pad_size_total, cudaMemcpyHostToDevice);
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
  	convolveC<<<gridSize, blockSize>>>(g_in, g_kern, g_out, pad_size_X, sum);
   	*t1 = timestamp();
  	cudaDeviceSynchronize();

  	// copy back the result array to the CPU
  	cudaMemcpy(out, g_out, sizeof(pixel_t)*data_size_X*data_size_Y, cudaMemcpyDeviceToHost);
    	cudaDeviceSynchronize();
            cudaFree(g_in);
    cudaFree(g_out);
    cudaFree(g_kern);
    return;
}
void cuda_function2(int data_size_X, int data_size_Y, float* kernel, float* in, float* out, double* t0, double* t1) {
    printf("Initiating CUDA...\n");
    

    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;
    int pad_size_X = data_size_X+2*kern_cent_X;
    int pad_size_Y = data_size_Y+2*kern_cent_Y;
    int pad_size_total = pad_size_Y * pad_size_X;

    // Padding code
    float kern_cpy[KERNX*KERNY];

    for (int i = 0; i<KERNX*KERNY; i++) {
        kern_cpy[i] = kernel[KERNX * KERNY - 1 - i];
    }
    float *in_cpy = (float*) calloc(pad_size_total, sizeof(float));

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
    convolveBW<<<gridSize, blockSize>>>(g_in, g_kern, g_out, pad_size_X, sum);
    *t1 = timestamp();
    cudaDeviceSynchronize();

    // copy back the result array to the CPU
    cudaMemcpy(out, g_out, sizeof(float)*data_size_X*data_size_Y, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
            cudaFree(g_in);
    cudaFree(g_out);
    cudaFree(g_kern);


  	return;
}
