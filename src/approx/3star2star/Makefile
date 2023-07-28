include ../../common.mk
VPATH += ../../common
all: 3star2star_omp_base 3star2star_edge_omp_base 3star2star_edge_sparse_args 3star2star_color_sparse_args

BIN=../../../bin
INCLUDES = -I../../../include

OBJS_ARGS = main_args.o graph.o VertexSet.o

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

3star2star_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

3star2star_edge_omp_base: $(OBJS) omp_base_3s2se.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base_3s2se.o -o $@ -lgomp
	mv $@ $(BIN)

3star2star_edge_sparse_args: $(OBJS_ARGS) 3star2star_edge_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS_ARGS) 3star2star_edge_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

3star2star_color_sparse_args: $(OBJS_ARGS) 3star2star_color_sparse_args.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS_ARGS) 3star2star_color_sparse_args.o -o $@ -lgomp
	mv $@ $(BIN)

