include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./cpu_kernels -I./gpu_kernels
all: clique_omp_base clique_gpu_base clique_multigpu

clique_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clique_omp_recursive: $(OBJS) omp_recursive.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_recursive.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clique_omp_iterative: $(OBJS) omp_iterative.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_iterative.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clique_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

clique_gpu_vp: $(OBJS) gpu_vp.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_vp.o -o $@ $(LIBS)
	mv $@ $(BIN)

clique_gpu_bitset: $(OBJS) gpu_bitset.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_bitset.o -o $@ $(LIBS)
	mv $@ $(BIN)

clique_gpu_base: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

clique_gpu_iterative: $(OBJS) gpu_iterative.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_iterative.o -o $@ $(LIBS)
	mv $@ $(BIN)

clique_multigpu: $(OBJS) multigpu.o scheduler.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) multigpu.o scheduler.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm -f *.o

.PHONY: clean

