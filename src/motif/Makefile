include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./cpu_kernels -I./gpu_kernels
all: motif_omp_base motif_omp_formula motif_gpu_base motif_gpu_formula motif_multigpu

motif_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

motif_omp_cmap: $(OBJS) omp_cmap.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_cmap.o -o $@ -lgomp
	mv $@ $(BIN)

motif_omp_formula: $(OBJS) omp_formula.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_formula.o -o $@ -lgomp
	mv $@ $(BIN)

motif_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

motif_gpu_base: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

motif_gpu_fission: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

motif_gpu_formula: $(OBJS) gpu_formula.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_formula.o -o $@ $(LIBS)
	mv $@ $(BIN)

motif_multigpu: $(OBJS) multigpu.o scheduler.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) multigpu.o scheduler.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
