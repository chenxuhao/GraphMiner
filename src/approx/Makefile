include ../common.mk
#VPATH += ../partitioner
all: tc_sample_omp_base tc_sample_edge_sparse tc_sample_color_sparse tc_sample_edge_stream tc_sample_edge_stream_v0 4cc_sample_omp_base 4cc_sample_edge_sparse 4cc_sample_color_sparse 4cc_sample_edge_stream 4cc_sample_edge_stream_v0

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


tc_sample_color_sparse: $(OBJS) tc_sample_color_sparse.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_sample_color_sparse.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_edge_stream: $(OBJS) tc_sample_edge_stream.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_sample_edge_stream.o -o $@ -lgomp
	mv $@ $(BIN)

tc_sample_edge_stream_v0: $(OBJS) tc_sample_edge_stream_v0.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) tc_sample_edge_stream_v0.o -o $@ -lgomp
	mv $@ $(BIN)

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


clean:
	rm *.o
