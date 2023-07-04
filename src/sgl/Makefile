include ../common.mk
INCLUDES+=-I./cpu_kernels -I./gpu_kernels
OBJS+=pattern.o
NVFLAGS+=--expt-extended-lambda
all: sgl_omp_base sgl_gpu_base sgl_multigpu

# baseline version, add make USE_SET_OPS=1 when make
sgl_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

# iterative version
sgl_omp_iterative: $(OBJS) omp_iterative.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_iterative.o -o $@ -lgomp
	mv $@ $(BIN)

# c-map version, default CPU version
sgl_omp_cmap: $(OBJS) omp_cmap.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_cmap.o -o $@ -lgomp
	mv $@ $(BIN)

# GraphZero version, add make USE_ZERO=1 when make
sgl_omp_zero: $(OBJS) omp_zero.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_zero.o -o $@ -lgomp
	mv $@ $(BIN)

sgl_omp_bj: $(OBJS) omp_bj.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_bj.o -o $@ -lgomp
	mv $@ $(BIN)

sgl_omp_lb: $(OBJS) omp_lb.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_lb.o -o $@ -lgomp
	mv $@ $(BIN)

# GPU edge parallel version, default GPU version
sgl_gpu_base: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

# GPU using binary join
sgl_gpu_bj: $(OBJS) gpu_bj.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_bj.o -o $@ $(LIBS)
	mv $@ $(BIN)

# GPU vertex parallel version
sgl_gpu_vp: $(OBJS) gpu_vp.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_vp.o -o $@ $(LIBS)
	mv $@ $(BIN)

# GPU counting, not listing, use edge parallel, add make COUNT=1 when make
sgl_gpu_count: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

sgl_multigpu: $(OBJS) multigpu.o scheduler.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) multigpu.o scheduler.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
