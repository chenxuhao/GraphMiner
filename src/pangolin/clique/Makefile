include ../common.mk
#INCLUDES+=-I $(CUB_DIR)
CXXFLAGS+=-std=c++11
all: kcl_base kcl_omp_base

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

kcl_base: $(OBJS) base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) base.o -o $@ $(LIBS)
	mv $@ $(BIN)

kcl_omp_base: $(OBJS) omp_base.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
