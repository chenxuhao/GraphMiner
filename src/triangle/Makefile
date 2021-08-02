include ../common.mk
INCLUDES+=-I./gpu_kernels
#INCLUDES+=-I$(CUB_DIR)
OBJS=main.o VertexSet.o
all: tc_omp_base tc_gpu_base tc_multigpu 

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

tc_multigpu: multigpu.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) multigpu.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
