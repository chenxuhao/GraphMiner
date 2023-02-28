
#clang++ cilk_base.cc -c  -I../../include -fopenmp=libiomp5 -I/home/cxh/OpenCilk/build/lib/clang/14.0.6/include/ -I/usr/lib/gcc/x86_64-linux-gnu/8/include
#clang++ -Wall -fopenmp=libiomp5 -std=c++11 -O3 -I../../include main.o VertexSet.o graph.o cilk_base.o -o tc_cilk_base -fopencilk

CILK_NWORKERS=20 ./tc_cilk_base ~/datasets/automine/livej/graph
