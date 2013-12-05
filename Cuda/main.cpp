#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include "readjpeg.h"
typedef struct
{
        float r;
        float g;
        float b;
} pixel_t;
void cuda_function(int data_size_X, int data_size_Y, float* kernel, pixel_t* in, pixel_t* out, double* t0, double* t1);
void cuda_function2(int data_size_X, int data_size_Y, float* kernel, float* in, float* out, double* t0, double* t1);

void normalize( float * kernel ) {
  int sum = 0;
  for (int i = 0; i < 25; i++ ) {
        sum += kernel[i];
  }
  for (int i = 0; i < 25 && sum != 0; i++ ) {
        kernel[i] /= sum;
  }
}


void print_matrix(float *array, int num, int x_size){
    int a = 0;
    printf("%5.2f, ", array[a]);
    for (a = 1; a < num; a++){
        if (a%x_size == x_size-1){
            printf("%5.2f,\n", array[a]);
        } else{
            printf("%5.2f, ", array[a]);
        }
    }
    printf("\n");
}

void convert_to_pixel(pixel_t *out, frame_ptr in)
{
        for(int y = 0; y < in->image_height; y++)
        {
                for(int x = 0; x < in->image_width; x++)
        {
                int r = (int)in->row_pointers[y][in->num_components*x + 0 ];
                int g = (int)in->row_pointers[y][in->num_components*x + 1 ];
                int b = (int)in->row_pointers[y][in->num_components*x + 2 ];
                out[y*in->image_width+x].r = (float)r;
                out[y*in->image_width+x].g = (float)g;
                out[y*in->image_width+x].b = (float)b;
         
        }
        }
}

void convert_to_frame(frame_ptr out, pixel_t *in)
{
        for(int y = 0; y < out->image_height; y++)
        {
                for(int x = 0; x < out->image_width; x++)
        {
                int r = (int)in[y*out->image_width + x].r;
                int g = (int)in[y*out->image_width + x].g;
                int b = (int)in[y*out->image_width + x].b;
                out->row_pointers[y][out->num_components*x + 0 ] = r;
                out->row_pointers[y][out->num_components*x + 1 ] = g;
                out->row_pointers[y][out->num_components*x + 2 ] = b;
        }
        }
}
#define KERNX 5 //this is the x-size of the kernel. It will always be odd.
#define KERNY 5 //this is the y-size of the kernel. It will always be odd.

int main(int argc, char *argv[]){

float kernel_0[] = { 0, 0, 0, 0, 0, // "sharpen"
                                         0, 0,-1, 0, 0,
                                         0,-1, 5,-1, 0,
                                         0, 0,-1, 0, 0,
                                         0, 0, 0, 0, 0, }; normalize(kernel_0);
float kernel_1[]={ 1, 1, 1, 1, 1, // blur
                                   1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, }; normalize(kernel_1);
float kernel_2[] = { 1, 1, 1, 1, 1, // weighted median filter
                                         2, 2, 2, 2, 2,
                                         3, 3, 3, 3, 3,
                                         2, 2, 2, 2, 2,
                                         1, 1, 1, 1, 1, };
float kernel_3[]={1,1,1,1,1, // weighted mean filter
                                  1,2,2,2,1,
                                  1,2,3,2,1,
                                  1,2,2,2,1,
                                  1,1,1,1,1, }; normalize(kernel_3);
float kernel_4[] = { 0, 0, 0, 0, 0, // "edge detect"
                                         0, 1, 0,-1, 0,
                                         0, 0, 0, 0, 0,
                                         0,-1, 0, 1, 0,
                                         0, 0, 0, 0, 0, };
float kernel_5[] = { 0, 0, 0, 0, 0, // "emboss"
                                         0,-2,-1, 0, 0,
                                         0,-1, 1, 1, 0,
                                         0, 0, 1, 2, 0,
                                         0, 0, 0, 0, 0, };
float kernel_6[] = {-1,-1,-1,-1,-1, // "edge detect"
                                        -1,-1,-1,-1,-1,
                                        -1,-1,24,-1,-1,
                                        -1,-1,-1,-1,-1,
                                        -1,-1,-1,-1,-1, };
float* kernels[7] = {kernel_0, kernel_1, kernel_2, kernel_3, kernel_4,
                                        kernel_5, kernel_6};

        int c;
        int color = 0;
        char *inName = NULL;
        char *outName = NULL;
        int width = -1, height = -1;
        int kernel_num = 1;
        frame_ptr frame;

        pixel_t *inPix = NULL;
        pixel_t *outPix = NULL;

        //grab command line arguments
    while((c = getopt(argc, argv, "i:k:o:c"))!=-1)
    {
            switch(c)
            {
            case 'i':
                    inName = optarg;
                    break;
            case 'o':
                    outName = optarg;
                    break;
            case 'k':
                    kernel_num = atoi(optarg);
                    break;
            case 'c':
                    color = 1;
            }
    }

    //input file name and output names
    inName = inName==0 ? (char*)"cpt-kurt.jpg" : inName;
    outName = outName==0 ? (char*)"output.jpg" : outName;

    //read file
    frame = read_JPEG_file(inName);
    if(!frame){
                   printf("unable to read %s\n", inName);
        exit(-1);
    }

    width = frame->image_width;
    height = frame->image_height;

    inPix = new pixel_t[width*height];
    outPix = new pixel_t[width*height];

    convert_to_pixel(inPix, frame);

    float* inFloats = new float[width*height];
    float* outFloats2 = new float[width*height];  
   for (int i=0; i<width*height; i++){
        outPix[i].r = 0;
        outPix[i].g = 0;
        outPix[i].b = 0;
        outFloats2[i] = 0;
        inFloats[i] = (inPix[i].r + inPix[i].g + inPix[i].b)/3;
        }
        float* kernel = kernels[kernel_num];

    double t0, t1;
    if(color == 1){
        cuda_function(width, height, kernel, inPix, outPix, &t0, &t1);
    } else {
        cuda_function2(width, height, kernel, inFloats, outFloats2, &t0, &t1);

    }
    printf("%g sec\n", t1-t0);
    if(color == 0) { 
        for (int i=0; i<width*height; i++){
            outPix[i].r = outFloats2[i];
            outPix[i].g = outFloats2[i];
            outPix[i].b = outFloats2[i];
        }
    } 
    convert_to_frame(frame, outPix);

    write_JPEG_file(outName,frame,75);
    destroy_frame(frame);

    delete [] inPix; 
    delete [] outPix;
    return 0;
}
