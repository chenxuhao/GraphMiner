include ../common.mk
INCLUDES+=-I./gpu_kernels -I$(TBB_HOME)/include
all: tc_omp_base tc_gpu_base tc_multigpu_base tc_multigpu

tc_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_omp_simd: $(OBJS) omp_simd.o intersect.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_simd.o intersect.o -o $@ -lgomp
	mv $@ $(BIN)

tc_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

tc_tbb_base: $(OBJS) tbb_base.o
	$(CXX) $(CXXFLAGS) -L$(TBB_HOME)/lib/intel64/gcc4.8 $(OBJS) tbb_base.o -o $@ -lgomp -ltbb
	mv $@ $(BIN)

tc_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_multigpu_base: $(OBJS) multigpu_base.o scheduler.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) multigpu_base.o scheduler.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_multigpu: $(OBJS) multigpu.o scheduler.o graph_partition.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) multigpu.o graph_partition.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_dist_cpu: $(OBJS) dist_cpu.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_cpu.o -o $@ -lgomp
	mv $@ $(BIN)

tc_dist_gpu: $(OBJS) dist_gpu.o gpu_kernel_wrapper.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_gpu.o gpu_kernel_wrapper.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
