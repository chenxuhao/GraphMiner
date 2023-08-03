include ../../common.mk
VPATH += ../../common
all: motif_omp_base motif_edge_sparse_args motif_color_sparse_args

BIN=../../../bin
INCLUDES = -I../../../include

OBJS_ARGS = main_args.o graph.o VertexSet.o

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

motif_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

motif_edge_sparse_args: $(OBJS_ARGS) motif_edge_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS_ARGS) motif_edge_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

motif_color_sparse_args: $(OBJS_ARGS) motif_color_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS_ARGS) motif_color_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

