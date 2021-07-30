include ../common.mk
INCLUDES+=-I $(CUB_DIR)
CXXFLAGS+=-std=c++11
SRC=main.cc
OBJS=omp_base.o base.o main.o
all: $(OBJS) motif_base motif_omp_base

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

motif_base: base.o main.o VertexSet.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o VertexSet.o -o $@ $(LIBS)
	mv $@ $(BIN)

motif_omp_base: omp_base.o main.o VertexSet.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o main.o VertexSet.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
