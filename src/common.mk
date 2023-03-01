DEBUG ?= 0
BIN = ../../bin/
CC := gcc
CXX := g++
ICC := $(ICC_HOME)/icc
ICPC := $(ICC_HOME)/icpc
MPICC := mpicc
MPICXX := mpicxx
NVCC := nvcc
#NVCC := $(CUDA_HOME)/bin/nvcc
CLANG := $(CILK_HOME)/bin/clang
CLANGXX := $(CILK_HOME)/bin/clang++

GENCODE_SM30 := -gencode arch=compute_30,code=sm_30
GENCODE_SM35 := -gencode arch=compute_35,code=sm_35
GENCODE_SM37 := -gencode arch=compute_37,code=sm_37
GENCODE_SM50 := -gencode arch=compute_50,code=sm_50
GENCODE_SM52 := -gencode arch=compute_52,code=sm_52
GENCODE_SM60 := -gencode arch=compute_60,code=sm_60
GENCODE_SM70 := -gencode arch=compute_70,code=sm_70
GENCODE_SM75 := -gencode arch=compute_75,code=sm_75
GENCODE_SM80 := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_SM86 := -gencode arch=compute_86,code=sm_86
CUDA_ARCH := $(GENCODE_SM70)
CXXFLAGS  := -Wall -fopenmp -std=c++17 -march=native
ICPCFLAGS := -O3 -Wall -qopenmp
NVFLAGS := $(CUDA_ARCH)
NVFLAGS += -Xptxas -v
NVFLAGS += -DUSE_GPU
NVLDFLAGS = -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcuda -lcudart
MPI_LIBS = -L$(MPI_HOME)/lib -lmpi
NVSHMEM_LIBS = -L$(NVSHMEM_HOME)/lib -lnvshmem -lnvToolsExt -lnvidia-ml -ldl -lrt
CILKFLAGS=-O3 -fopenmp=libiomp5 -fopencilk

ifeq ($(VTUNE), 1)
	CXXFLAGS += -g
endif
ifeq ($(NVPROF), 1)
	NVFLAGS += -lineinfo
endif

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G
else
	CXXFLAGS += -O3
	NVFLAGS += -O3 -w
endif

INCLUDES := -I../../include
LIBS := $(NVLDFLAGS) -lgomp
CILK_INC=-I$(GCC_HOME)/include -I$(CILK_CLANG)/include

ifeq ($(PAPI), 1)
CXXFLAGS += -DENABLE_PAPI
INCLUDES += -I$(PAPI_HOME)/include
LIBS += -L$(PAPI_HOME)/lib -lpapi
endif

ifeq ($(USE_TBB), 1)
LIBS += -L$(TBB_HOME)/lib/intel64/gcc4.8/ -ltbb
endif

VPATH += ../common
OBJS=main.o VertexSet.o graph.o

ifneq ($(NVSHMEM),)
CXXFLAGS += -DUSE_MPI
NVFLAGS += -DUSE_NVSHMEM -DUSE_MPI -dc
endif

# CUDA vertex parallel
ifneq ($(VPAR),)
NVFLAGS += -DVERTEX_PAR
endif

# CUDA CTA centric
ifneq ($(CTA),)
NVFLAGS += -DCTA_CENTRIC
endif

ifneq ($(PROFILE),)
CXXFLAGS += -DPROFILING
endif

ifneq ($(USE_SET_OPS),)
CXXFLAGS += -DUSE_MERGE
endif

ifneq ($(USE_SIMD),)
CXXFLAGS += -DSI=0
endif

# counting or listing
ifneq ($(COUNT),)
NVFLAGS += -DDO_COUNT
endif

# GPU vertex/edge parallel 
ifeq ($(VERTEX_PAR),)
  NVFLAGS += -DEDGE_PAR
else
  NVFLAGS += -DVERTEX_PAR
endif

# CUDA unified memory
ifneq ($(USE_UM),)
NVFLAGS += -DUSE_UM
endif

# kernel fission
ifneq ($(FISSION),)
NVFLAGS += -DFISSION
endif

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

%.o: %.cxx
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) -c $<

