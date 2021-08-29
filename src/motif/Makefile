include ../common.mk
#INCLUDES+=-I$(CUB_DIR) 
INCLUDES+=-I./cpu_kernels -I./gpu_kernels
all: motif_omp_base

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

motif_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

motif_gpu_base: $(OBJS) gpu_base.o 
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
