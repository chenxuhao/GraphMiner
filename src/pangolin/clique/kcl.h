// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

//#include "common.h"
#include "graph.h"
/*
GARDENIA Benchmark Suite
Kernel: k-Cliques Listing (k-CL)
Author: Xuhao Chen

Will count the number of cliques of size k 

Requires input graph:
  - to be undirected
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

This implementation reduces the search space by counting 
each clique only once. This implementation counts cliques 
in a directed acyclic graph (DAG).

kcl_omp : one thread per vertex using OpenMP
kcl_base: one thread per vertex using CUDA
kcl_warp: one warp per vertex using CUDA
*/

#define MAX_SIZE 5
void KclSolver(Graph &g, unsigned k, uint64_t &total);
void KclVerifier(Graph &g, unsigned k, uint64_t test_total);

