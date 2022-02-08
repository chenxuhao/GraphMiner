CUDA_HOME=/usr/local/cuda
CUB_DIR=../../../../cub
BIN=../../../bin/pangolin
CC=gcc
CXX=g++
NVCC=nvcc
CUDA_ARCH := -gencode arch=compute_70,code=sm_70
CXXFLAGS=-Wall -fopenmp -std=c++11
ICPCFLAGS=-O3 -Wall -qopenmp
NVFLAGS=$(CUDA_ARCH)
#NVFLAGS+=-Xptxas -v -lineinfo
ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G
else
	CXXFLAGS += -g -O3
	NVFLAGS += -O3 -w
endif
LIBS += -lgomp
INCLUDES = -I../../../include
VPATH=../../common
OBJS=graph.o VertexSet.o main.o
