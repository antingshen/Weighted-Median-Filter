#include <sys/time.h>
#include <time.h>

int serial(float* in, float* out, int data_size_X, 
	int data_size_Y, float* kernel, struct timeval *start, 
    struct timeval *end);

int openCL(float* in, float* out, int data_size_X, 
	int data_size_Y, float* kernel, struct timeval *start, 
    struct timeval *end);

int cuda(float* in, float* out, int data_size_X, 
	int data_size_Y, float* kernel, struct timeval *start, 
    struct timeval *end);

