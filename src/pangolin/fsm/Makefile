include ../common.mk
#INCLUDES+=-I $(CUB_DIR)
CXXFLAGS+=-std=c++11
all: fsm_base fsm_omp_base

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

fsm_base: $(OBJS) base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) base.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

fsm_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
