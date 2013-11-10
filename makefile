UNAME = $(shell uname)

ifeq ($(UNAME),Linux)

endif

ifeq ($(UNAME),Darwin)

endif

CC = g++

CFLAGS = -wall -O3 -pipe -fno-omit-frame-pointer

all:	test-all

test-all: test.o serial.o openCL.o cuda.o
	$(CC) -o $@ $(LIBS) test.o serial.o openCL.o cuda.o $(GOTOLIB)

%.o: %.c
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f *~ test *.o
