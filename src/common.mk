DEBUG ?= 0
#CUDA_HOME=/opt/apps/cuda/11.3
CUDA_HOME=/usr/local/cuda
PAPI_HOME=/usr/local/papi-6.0.0
ICC_HOME=/opt/intel/compilers_and_libraries/linux/bin/intel64
MKLROOT=/opt/intel/mkl
CUB_DIR=../../cub
MGPU_DIR=../../moderngpu
BIN=../../bin/
CC=gcc
CXX=g++
ICC=$(ICC_HOME)/icc
ICPC=$(ICC_HOME)/icpc
MPICC=mpicc
MPICXX=mpicxx
NVCC=nvcc
#CUDA_ARCH := -gencode arch=compute_75,code=sm_75
CUDA_ARCH := -gencode arch=compute_70,code=sm_70
CXXFLAGS=-Wall -fopenmp -std=c++11 -march=native
ICPCFLAGS=-O3 -Wall -qopenmp
NVFLAGS=$(CUDA_ARCH)
#NVFLAGS+=-Xptxas -v
NVFLAGS+=-DUSE_GPU

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
INCLUDES = -I../../include
LIBS=-L$(CUDA_HOME)/lib64 -lcudart -lgomp

ifeq ($(PAPI), 1)
CXXFLAGS += -DENABLE_PAPI
INCLUDES += -I$(PAPI_HOME)/include
LIBS += -L$(PAPI_HOME)/lib -lpapi
endif

ifeq ($(USE_TBB), 1)
LIBS += -L/h2/xchen/work/gardenia_code/tbb2020/lib/intel64/gcc4.8/ -ltbb
endif

VPATH += ../common
OBJS=main.o VertexSet.o graph.o

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
endif

# CUDA unified memory
ifneq ($(USE_UM),)
NVFLAGS += -DUSE_UM
endif

# kernel fission
ifneq ($(FISSION),)
NVFLAGS += -DFISSION
endif

