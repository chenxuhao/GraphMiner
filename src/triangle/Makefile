include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./gpu_kernels
all: tc_omp_base tc_gpu_base tc_multigpu 

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $<

tc_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_omp_simd: $(OBJS) omp_simd.o intersect.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_simd.o intersect.o -o $@ -lgomp
	mv $@ $(BIN)

tc_gpu_base: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_cta: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_vp: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_multigpu: $(OBJS) multigpu.o scheduler.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) multigpu.o scheduler.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_bs: $(OBJS) bin_search.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) bin_search.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_hindex: $(OBJS) hindex.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) hindex.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_merge: $(OBJS) merge.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) merge.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_dist_cpu: $(OBJS) dist_cpu.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_cpu.o -o $@ -lgomp
	mv $@ $(BIN)

tc_dist_gpu: $(OBJS) dist_gpu.o gpu_kernel_wrapper.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_gpu.o gpu_kernel_wrapper.o -o $@ -lgomp -lcudart
	mv $@ $(BIN)

clean:
	rm *.o
