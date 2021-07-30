include ../common.mk
INCLUDES+=-I $(CUB_DIR)
CXXFLAGS+=-std=c++11
SRC=main.cc
OBJS=main.o VertexSet.o
all: fsm_base fsm_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

ifeq ($(HOST),X86)
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<
else
%.o: %.cc
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<
endif

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

fsm_base: base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) base.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

fsm_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
