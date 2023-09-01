include ../../common.mk
INCLUDES := -I../../../include
BIN=../../../bin/
VPATH := ../$(VPATH)
INCLUDES+=-I./gpu_kernels -I$(TBB_HOME)/include -I$(OPENBLAS_DIR)/include
CLANGXX=/home/azliu/build/bin/clang++
all: tc_omp_mm

tc_omp_mm: $(OBJS) omp_mm.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_mm.o -o $@ -lgomp -L$(OPENBLAS_DIR)/lib -lopenblas 
	mv $@ $(BIN)

clean:
	rm *.o
