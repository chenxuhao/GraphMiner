include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./cpu_kernels -I./gpu_kernels
OBJS=main.o VertexSet.o
all: sgl_omp_base

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

# baseline version, add make USE_SET_OPS=1 when make
sgl_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

# GraphZero version, add make USE_ZERO=1 when make
sgl_omp_zero: $(OBJS) omp_zero.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_zero.o -o $@ -lgomp
	mv $@ $(BIN)

# GPU edge parallel version, default GPU version
sgl_gpu_base: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

sgl_multigpu: $(OBJS) multigpu.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) multigpu.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
