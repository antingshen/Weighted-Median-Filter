#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

#include "readjpeg.h"

void normalize( float * kernel ) {
  int sum = 0;
  for (int i = 0; i < 25; i++ ) {
	sum += kernel[i];
  }
  for (int i = 0; i < 25 && sum != 0; i++ ) {
	kernel[i] /= sum;
  }
}

typedef struct
{
	float r;
	float g;
	float b;
} pixel_t;

double timestamp()
{
	struct timeval tv;
	gettimeofday (&tv, 0);
	return tv.tv_sec + 1e-6*tv.tv_usec;
}

void blur_frame(int width, int height, float* kernel, pixel_t *in, pixel_t *out){
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
int conv2D(int data_size_X, int data_size_Y,
					float* kernel, pixel_t* in, pixel_t* out)
{
	// the x coordinate of the kernel's center
	int kern_cent_X = (KERNX - 1)/2;
	// the y coordinate of the kernel's center
	int kern_cent_Y = (KERNY - 1)/2;
	
	// main convolution loop
	for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
		for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
			for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
					// only do the operation if not out of bounds
					if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						//Note that the kernel is flipped
						out[x+y*data_size_X].r += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X].r;
						out[x+y*data_size_X].g += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X].g;
						out[x+y*data_size_X].b += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X].b;
					}
				}
			}
		}
	}

	for (int i=0; i<data_size_X*data_size_Y; i++){
		if (out[i].r<0){
			out[i].r=0;
		}
		if (out[i].r>255){
			out[i].r=255;
		}
		if (out[i].g<0){
			out[i].g=0;
		}
		if (out[i].g>255){
			out[i].g=255;
		}
		if (out[i].b<0){
			out[i].b=0;
		}
		if (out[i].b>255){
			out[i].b=255;
		}
	}
	return 1;
}

int main(int argc, char *argv[])
{
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
float kernel_2[] = { 0, 0, 0, 0, 0, // darken
					 0, 0, 0, 0, 0,
					 0, 0,0.5, 0, 0,
					 0, 0, 0, 0, 0,
					 0, 0, 0, 0, 0, };
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
	char *inName = NULL;
	char *outName = NULL;
	int width=-1,height=-1;
	int kernel_num = 1;
	frame_ptr frame;

	pixel_t *inPix=NULL;
	pixel_t *outPix=NULL;

	while((c = getopt(argc, argv, "i:k:o:"))!=-1)
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
		}
	}

	inName = inName==0 ? (char*)"cpt-kurt.jpg" : inName;
	outName = outName==0 ? (char*)"output.jpg" : outName;

	frame = read_JPEG_file(inName);
	if(!frame)
	{
		printf("unable to read %s\n", inName);
		exit(-1);
	}

	width = frame->image_width;
	height = frame->image_height;
 
	inPix = new pixel_t[width*height];
	outPix = new pixel_t[width*height];

	for (int i=0; i<width*height; i++){
		outPix[i].r = 0;
		outPix[i].g = 0;
		outPix[i].b = 0;
	}
	
	convert_to_pixel(inPix, frame);

	float* kernel = kernels[kernel_num];

	double t0 = timestamp();
	conv2D(width, height, kernel, inPix, outPix);
	t0 = timestamp() - t0;
	printf("%g sec\n", t0);

	convert_to_frame(frame, outPix);

	write_JPEG_file(outName,frame,75);
	destroy_frame(frame);

	delete [] inPix; 
	delete [] outPix;
	return 0;
}
