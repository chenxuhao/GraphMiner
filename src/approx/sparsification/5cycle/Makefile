include ../../common.mk
VPATH += ../../common
all: 5cycle_sample_perf_args

BIN=../../../bin
INCLUDES = -I../../../include


%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

5cycle_sample_perf_args: main_sample_args.o graph.o VertexSet.o 5cycle_sample_perf_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o 5cycle_sample_perf_args.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
