#CXXFLAGS=-Wall -std=gnu++0x -fopenmp -O3
#CXXDEBUGFLAGS=-Wall -std=gnu++0x -fopenmp -g

include ../../common.mk
VPATH += ../../common
all: colorcoding_omp_base

BIN=../../../bin
INCLUDES = -I../../../include

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

colorcoding_omp_base: main.o graph.o VertexSet.o
	${CXX} ${CXXFLAGS} main.o graph.o VertexSet.o -o $@ -lgomp

clean:
	rm -f *.o
