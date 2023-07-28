include ../../common.mk
VPATH += ../../common
all: 3star_omp_base 3star_edge_sparse_args 3star_color_sparse_args tailedtriangle_omp_base

BIN=../../../bin
INCLUDES = -I../../../include

OBJS_ARGS = main_args.o graph.o VertexSet.o

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

3star_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

3star_edge_sparse_args: $(OBJS_ARGS) 3star_edge_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS_ARGS) 3star_edge_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

3star_color_sparse_args: $(OBJS_ARGS) 3star_color_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS_ARGS) 3star_color_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

tailedtriangle_omp_base: $(OBJS_ARGS) omp_base_tailed_triangle.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base_tailed_triangle.o -o $@ -lgomp
	mv $@ $(BIN)
