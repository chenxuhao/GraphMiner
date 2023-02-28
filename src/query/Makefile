include ../common.mk
INCLUDES+=-I./cpu_kernels -I./gpu_kernels
OBJS+=pattern.o
all: query_omp_base query_omp_auxiliary

query_gpu_base: gpu_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

query_omp_base: omp_base.o filter.o query_plan.o intersect.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o filter.o query_plan.o intersect.o -o $@ -lgomp
	mv $@ $(BIN)

query_omp_auxiliary: omp_auxiliary.o filter.o query_plan.o intersect.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_auxiliary.o filter.o query_plan.o intersect.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
