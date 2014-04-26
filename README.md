Weighted-Median-Filter
======================

### Running the Serial and OpenMP code

-i Specifies input image  
-o Specifies output image  
-k Specifies the kernel to run  

To run with default input image and output.jpg:
```
$ ./conv2d -k 2
```

or run with custom input/output images:
```
$ ./conv2d -i input.jpg -o output.jpg -k 2
```

### Running the OpenCL code

-g Specifies output image to be in grayscale  
By default, output is in color

```
$ make clean && make
$ ./conv2d -i input.jpg -o output.jpg -k 2
```

### Running the CUDA code

-c Specifies output image to be in color  
By default, output is in grayscale  
First run the script to compile the C++ and CUDA code.

```
$ ./script.sh
$ ./main -i input.jpg -o output.jpg -k 2 -c
```

### List of Available Kernels
```
0. Sharpen
1. Uniform
2. Weighted Median
3. Weighted Mean
4. Gaussian
5. Emboss
6. Edge Detect
```
