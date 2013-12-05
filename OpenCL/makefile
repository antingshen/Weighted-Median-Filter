OBJS=conv2d.o readjpeg.o clhelp.o
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Linux)
OCL_INC=/usr/local/cuda-4.2/include
OCL_LIB=/usr/local/cuda-4.2/lib64

%.o: %.cpp readjpeg.h clhelp.h
	g++ -O2 -c $< -I$(OCL_INC)

all: $(OBJS)
	g++ conv2d.o readjpeg.o clhelp.o -o conv2d -L$(OCL_LIB) -lOpenCL -ljpeg
endif

ifeq ($(UNAME_S), Darwin)
%.o: %.cpp readjpeg.h clhelp.h
	g++ -O2 -c $<

all: $(OBJS)
	g++ conv2d.o readjpeg.o clhelp.o -o conv2d -framework OpenCL -ljpeg
endif

clean:
	rm -rf conv2d $(OBJS)