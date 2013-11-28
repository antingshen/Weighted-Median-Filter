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


typedef struct
{
	float r;
	float g;
	float b;
} pixel_t;

// double timestamp()
// {
// 	struct timeval tv;
// 	gettimeofday (&tv, 0);
// 	return tv.tv_sec + 1e-6*tv.tv_usec;
// }

void blur_frame(int width, int height, float* kernel, pixel_t *in, pixel_t *out){
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
int conv2D(int data_size_X, int data_size_Y, float* kernel, float* in, float* out, double* t0, double* t1)
{
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

    // OpenCL setup
    std::string convolve_kernel_str;

    std::string convolve_name_str = std::string("convolve");
    std::string convolve_kernel_file = std::string("conv2d.cl");

    cl_vars_t cv; 
    cl_kernel convolve;

    readFile(convolve_kernel_file,
    convolve_kernel_str);

    initialize_ocl(cv);

    compile_ocl_program(convolve, cv, convolve_kernel_str.c_str(),
    convolve_name_str.c_str());

    cl_int err = CL_SUCCESS;

    cl_mem g_in, g_out, g_kern;

    g_in = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int)*pad_size_total,NULL,&err);
    CHK_ERR(err);  

    g_kern = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int)*KERNY*KERNX,NULL,&err);
    CHK_ERR(err);

    g_out = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int)*data_size_X*data_size_Y,NULL,&err);
    CHK_ERR(err);

    err = clEnqueueWriteBuffer(cv.commands, g_in, true, 0, sizeof(int)*pad_size_total,
        in_cpy, 0, NULL, NULL);
    CHK_ERR(err);

    err = clEnqueueWriteBuffer(cv.commands, g_kern, true, 0, sizeof(int)*KERNX*KERNY,
        kern_cpy, 0, NULL, NULL);
    CHK_ERR(err);

    size_t global_work_size[2] = {data_size_X, data_size_Y};

    err = clSetKernelArg(convolve,0,
        sizeof(cl_mem), &g_in);
    CHK_ERR(err);

    err = clSetKernelArg(convolve,1,
        sizeof(cl_mem), &g_kern);
    CHK_ERR(err);

    err = clSetKernelArg(convolve,2,
        sizeof(cl_mem), &g_out);
    CHK_ERR(err);

    err = clSetKernelArg(convolve,3,
        sizeof(int), &pad_size_X);
    CHK_ERR(err);

    // int kern_width = KERNX;
    // err = clSetKernelArg(convolve,4,
    //     sizeof(int), &kern_width);
    // CHK_ERR(err);

    float sum = 0;
    for (int i=0; i<KERNX*KERNY; i++){
    	sum += kernel[i];
    }
    sum /= 2;
    err = clSetKernelArg(convolve,4,
    	sizeof(float), &sum);
    CHK_ERR(err);

    *t0 = timestamp();
    err = clEnqueueNDRangeKernel(cv.commands, 
        convolve,
        2,//work_dim,
        NULL, //global_work_offset
        global_work_size, //global_work_size
        NULL, //local_work_size
        0, //num_events_in_wait_list
        NULL, //event_wait_list
        NULL //
        );
    *t1 = timestamp();
    CHK_ERR(err);

    err = clEnqueueReadBuffer(cv.commands, g_out, true, 0, sizeof(int)*data_size_X*data_size_Y,
        out, 0, NULL, NULL);
    CHK_ERR(err);

    clReleaseMemObject(g_in);
    clReleaseMemObject(g_out);
    clReleaseMemObject(g_kern);

    uninitialize_ocl(cv);

    // printf("--KERNEL--\n");
    // print_matrix(kern_cpy, KERNX*KERNY, KERNX);
    // printf("--INPUT--\n");
    // print_matrix(in, data_size_X*data_size_Y, data_size_X);
    // printf("--MY OUTPUT--\n");
    // print_matrix(out, data_size_X*data_size_Y, data_size_X);

    return 1;

}

int main(int argc, char *argv[])
{
float kernel_0[] = { 0, 0, 0, 0, 0, // "sharpen"
					 0, 0,-1, 0, 0,
					 0,-1, 5,-1, 0,
					 0, 0,-1, 0, 0,
					 0, 0, 0, 0, 0, };
float kernel_1[]={ 1, 1, 1, 1, 1, // blur
				   1, 1, 1, 1, 1,
				   1, 1, 1, 1, 1,
				   1, 1, 1, 1, 1,
				   1, 1, 1, 1, 1, };
float kernel_2[] = { 0, 0, 0, 0, 0, // darken
					 0, 0, 0, 0, 0,
					 0, 0,0.5, 0, 0,
					 0, 0, 0, 0, 0,
					 0, 0, 0, 0, 0, };
float kernel_3[]={1,1,1,1,1, // weighted mean filter
				  1,2,2,2,1,
				  1,2,3,2,1,
				  1,2,2,2,1,
				  1,1,1,1,1, };
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

	convert_to_pixel(inPix, frame);

	float* inFloats = new float[width*height];
	float* outFloats = new float[width*height];

	for (int i=0; i<width*height; i++){
		outPix[i].r = 0;
		outPix[i].g = 0;
		outPix[i].b = 0;
		outFloats[i] = 0;
		inFloats[i] = (inPix[i].r + inPix[i].g + inPix[i].b)/3;
	}

	float* kernel = kernels[kernel_num];

	double t0, t1;
	conv2D(width, height, kernel, inFloats, outFloats, &t0, &t1);
	printf("%g sec\n", t1-t0);

	for (int i=0; i<width*height; i++){
		outPix[i].r = outFloats[i];
		outPix[i].g = outFloats[i];
		outPix[i].b = outFloats[i];
	}

	convert_to_frame(frame, outPix);

	write_JPEG_file(outName,frame,75);
	destroy_frame(frame);

	delete [] inPix; 
	delete [] outPix;
	return 0;
}
