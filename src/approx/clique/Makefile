include ../../common.mk
VPATH += ../../common
all: clique_sample_omp_base clique_sample_color_sparse clique_sample_edge_sparse clique4_omp_base clique_omp_sample_perf clique4_sample_perf clique4_sample_perf_args clique4_sample_edge_sparse

BIN=../../../bin
INCLUDES = -I../../../include

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<



clique_sample_omp_base: $(OBJS) clique_omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) clique_omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

clique_sample_edge_sparse: $(OBJS) clique_sample_edge_sparse.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) clique_sample_edge_sparse.o -o $@ -lgomp
	mv $@ $(BIN)


clique_sample_color_sparse: $(OBJS) clique_sample_color_sparse.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) clique_sample_color_sparse.o -o $@ -lgomp
	mv $@ $(BIN)

clique4_omp_base: $(OBJS) clique4_omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) clique4_omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

clique_omp_sample_perf: $(OBJS) clique_omp_sample_perf.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) clique_omp_sample_perf.o -o $@ -lgomp
	mv $@ $(BIN)

clique4_sample_perf: $(OBJS) clique4_sample_perf.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) clique4_sample_perf.o -o $@ -lgomp
	mv $@ $(BIN)

clique4_sample_perf_args: main_sample_args.o graph.o VertexSet.o clique4_sample_perf_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o clique4_sample_perf_args.o -o $@ -lgomp
	mv $@ $(BIN)

clique4_sample_edge_sparse: main_sample_args.o graph.o VertexSet.o clique4_sample_edge_sparse.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main_sample_args.o graph.o VertexSet.o clique4_sample_edge_sparse.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
