include ../../common.mk
INCLUDES := -I../../../include
BIN=../../../bin/
VPATH := ../$(VPATH)
INCLUDES+=-I./cpu_kernels -I./gpu_kernels -I$(OPENBLAS_DIR)/include
CXXFLAGS+=-pie
all: clique_omp_mm

diamond_omp_mm: $(OBJS) omp_diamond_mm.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_diamond_mm.o -o $@ -lgomp -L$(OPENBLAS_DIR)/lib -lopenblas 
	mv $@ $(BIN)

clique_omp_mm: $(OBJS) omp_clique_mm.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_clique_mm.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm -f *.o

.PHONY: clean

