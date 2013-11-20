#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
          int index = blockIdx.x + blockIdx.y*numCols;
        uchar4 pixel = rgbaImage[index];
        greyImage[index] = (pixel.x*0.299f) + (pixel.y*0.587f) + (pixel.z*0.114f);

}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{

  const dim3 blockSize(1, 1, 1);
  const dim3 gridSize(numCols, numRows, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
}