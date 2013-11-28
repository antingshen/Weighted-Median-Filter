OBJS=conv2d.o readjpeg.o

%.o: %.cpp readjpeg.h
	g++ -O2 -c $<

all: $(OBJS)
	g++ -O2 $(OBJS) -o conv2d -ljpeg

clean:
	rm -rf conv2d $(OBJS)