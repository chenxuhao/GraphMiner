include ../common.mk
INCLUDES+=-I$(CUB_DIR) -I./gpu_kernels
OBJS=main.o verifier.o VertexSet.o
all: tc_omp_base tc_gpu_base tc_multigpu 

# CUDA vertex parallel
ifneq ($(VPAR),)
NVFLAGS += -DVERTEX_PAR
endif

# CUDA CTA centric
ifneq ($(CTA),)
NVFLAGS += -DCTA_CENTRIC
endif

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

tc_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_gpu_base: gpu_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_cta: gpu_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_vp: gpu_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_multigpu: multigpu.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) multigpu.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_bs: bin_search.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) bin_search.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_hindex: hindex.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) hindex.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_gpu_merge: merge.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) merge.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
