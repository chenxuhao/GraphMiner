include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./cpu_kernels -I./gpu_kernels
OBJS=main.o VertexSet.o
all: clique_omp_base 

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

clique_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) omp_base.o $(OBJS) -o $@
	mv $@ $(BIN)

clique_gpu_base: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm -f *.o

.PHONY: clean

