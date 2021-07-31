DEBUG ?= 0
CUDA_HOME=/usr/local/cuda
CUB_DIR=../../cub
BIN=../../bin/
CC=gcc
CXX=g++
NVCC=nvcc
CUDA_ARCH := -gencode arch=compute_70,code=sm_70
CXXFLAGS=-Wall -fopenmp -std=c++11
ICPCFLAGS=-O3 -Wall -qopenmp
NVFLAGS=$(CUDA_ARCH)
#NVFLAGS+=-Xptxas -v
NVFLAGS+=-DUSE_GPU
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
LIBS += -lgomp
VPATH += ../common

