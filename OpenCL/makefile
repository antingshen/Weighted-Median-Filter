OBJS=conv2d.o readjpeg.o clhelp.o

%.o: %.cpp readjpeg.h clhelp.h
	g++ -O2 -c $<

all: $(OBJS)
	g++ conv2d.o readjpeg.o clhelp.o -o conv2d -framework OpenCL -ljpeg

clean:
	rm -rf conv2d $(OBJS)