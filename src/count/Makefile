include ../common.mk
INCLUDES+=-I./cpu_kernels -I./gpu_kernels
OBJS+=pattern.o
NVFLAGS+=--expt-extended-lambda
all: count_omp_base 

count_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

count_omp_bj: $(OBJS) omp_bj.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_bj.o -o $@ -lgomp
	mv $@ $(BIN)

count_gpu_base: $(OBJS) gpu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

count_gpu_bj: $(OBJS) gpu_bj.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_bj.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
