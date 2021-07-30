// Copyright 2021 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
/*
GARDENIA Benchmark Suite
Kernel: Frequent Subgraph Mining (FSM)

Will count the number of frequent patterns in an undirected graph 

Requires input graph:
  - to be undirected graph
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

fsm_base: one thread per vertex using CUDA
fsm_warp: one warp per vertex using CUDA
*/
#define MAX_SIZE 5
#define MAX_NUM_PATTERNS 21251
void FsmSolver(Graph &g, unsigned k, unsigned minsup, int nlabels, int &total_num);
void FsmVerifier(Graph &g, unsigned k, unsigned minsup);
