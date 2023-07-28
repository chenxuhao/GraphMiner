include ../../common.mk
VPATH += ../../common
all: house_sample_color_sparse_args

BIN=../../../bin
INCLUDES = -I../../../include


%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

house_sample_color_sparse_args: main_sample_args.o graph.o VertexSet.o house_sample_color_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o house_sample_color_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)


clean:
	rm *.o
