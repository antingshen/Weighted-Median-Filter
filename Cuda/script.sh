rm *.o
rm main
rm output.jpg

g++ -c main.cpp readjpeg.cpp
nvcc -c cudafun.cu
nvcc -o main main.o cudafun.o readjpeg.o -ljpeg
