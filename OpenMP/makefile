OBJS=conv2d.o readjpeg.o

%.o: %.cpp readjpeg.h
	g++ -fopenmp -O2 -c $<

all: $(OBJS)
	g++ -fopenmp -O2 $(OBJS) -o conv2d -ljpeg

clean:
	rm -rf conv2d $(OBJS)