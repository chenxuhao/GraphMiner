include ../common.mk
#VPATH += ../partitioner
all: tc_sample_omp_base tc_sample_edge_sparse tc_sample_color_sparse tc_sample_color_sparse_args tc_sample_mask tc_sample_edge_stream tc_sample_edge_stream_v0 tc_sample_edge_mask tc_sample_perf_args tc_sample_edge_sparse_args tc_sample_edge_stream_args 

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

tc_sample_omp_base: $(OBJS) tc_omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_edge_sparse: $(OBJS) tc_sample_edge_sparse.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_sample_edge_sparse.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_mask: $(OBJS) tc_sample_mask.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_sample_mask.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_edge_mask: $(OBJS) tc_sample_edge_mask.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_sample_edge_mask.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_color_sparse: $(OBJS) tc_sample_color_sparse.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_sample_color_sparse.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_edge_stream: $(OBJS) tc_sample_edge_stream.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_sample_edge_stream.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_edge_stream_v0: $(OBJS) tc_sample_edge_stream_v0.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_sample_edge_stream_v0.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_perf_args: main_sample_args.o graph.o VertexSet.o tc_sample_perf_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o tc_sample_perf_args.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_edge_sparse_args: main_sample_args.o graph.o VertexSet.o tc_sample_edge_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o tc_sample_edge_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

4cc_sample_edge_sparse_args: main_sample_args.o graph.o VertexSet.o 4cc_sample_edge_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o 4cc_sample_edge_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

4cc_sample_edge_stream_args: main_sample_args.o graph.o VertexSet.o 4cc_sample_edge_stream_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o 4cc_sample_edge_stream_args.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_edge_stream_args: main_sample_args.o graph.o VertexSet.o tc_sample_edge_stream_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o tc_sample_edge_stream_args.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_color_sparse_args: main_sample_args.o graph.o VertexSet.o tc_sample_color_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o tc_sample_color_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

tc_ASAP_args: main_sample_args.o graph.o ASAP.o VertexSet.o tc_ASAP_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o ASAP.o VertexSet.o tc_ASAP_args.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
