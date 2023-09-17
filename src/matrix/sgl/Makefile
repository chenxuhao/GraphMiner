include ../../common.mk
INCLUDES := -I../../../include
BIN=../../../bin/
VPATH := ../$(VPATH)
INCLUDES+=-I$(MGPU_DIR)
INCLUDES+=-I./cpu_kernels -I./gpu_kernels 
INCLUDES+=-I$(OPENBLAS_DIR)/include
CXXFLAGS+=-pie
OBJS+=pattern.o
NVFLAGS+=--expt-extended-lambda
all: sgl_omp_mm

sgl_omp_mm: $(OBJS) omp_mm.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_mm.o -o $@ -lgomp -L$(OPENBLAS_DIR)/lib -lopenblas 
	mv $@ $(BIN)

clean:
	rm *.o
