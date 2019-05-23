.PHONY: all clean

CFLAGS+= -std=c++11 -Wall
#CFLAGS+= -g
CFLAGS+= -O3
LDFLAGS+=-lm     # link to math library

TARGET=train test

all: $(TARGET)
# type make/make all to compile test_hmm

hmm: hmm.cpp hmm.h 
	g++ -o hmm.o hmm.cpp $(CFLAGS)

train: train.cpp hmm.o 
	g++ -o $@ train.cpp hmm.o $(CFLAGS)

test: test.cpp hmm.o
	g++ -o $@ test.cpp hmm.o $(CFLAGS)

clean:
	$(RM) $(TARGET) model_0* result.txt acc.txt # type make clean to remove the compiled file
