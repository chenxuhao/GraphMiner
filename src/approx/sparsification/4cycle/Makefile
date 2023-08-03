include ../../common.mk
VPATH += ../../common
all: 4cc_sample_omp_base 4cc_sample_edge_sparse 4cc_sample_color_sparse 4cc_sample_edge_stream 4cc_sample_edge_stream_v0 4cc_sample_edge_stream_args 4cc_sample_edge_sparse_args

BIN=../../../bin
INCLUDES = -I../../../include

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

4cc_sample_omp_base: $(OBJS) 4c_omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) 4c_omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

4cc_sample_edge_sparse: $(OBJS) 4cc_sample_edge_sparse.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) 4cc_sample_edge_sparse.o -o $@ -lgomp
	mv $@ $(BIN)

4cc_sample_color_sparse: $(OBJS) 4cc_sample_color_sparse.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) 4cc_sample_color_sparse.o -o $@ -lgomp
	mv $@ $(BIN)

4cc_sample_edge_stream: $(OBJS) 4cc_sample_edge_stream.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) 4cc_sample_edge_stream.o -o $@ -lgomp
	mv $@ $(BIN)

4cc_sample_edge_stream_v0: $(OBJS) 4cc_sample_edge_stream_v0.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) 4cc_sample_edge_stream_v0.o -o $@ -lgomp
	mv $@ $(BIN)

4cc_sample_edge_sparse_args: main_sample_args.o graph.o VertexSet.o 4cc_sample_edge_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o 4cc_sample_edge_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

4cc_sample_edge_stream_args: main_sample_args.o graph.o VertexSet.o 4cc_sample_edge_stream_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o 4cc_sample_edge_stream_args.o -o $@ -lgomp
	mv $@ $(BIN)


clean:
	rm *.o
