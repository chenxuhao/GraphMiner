// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
/*
GARDENIA Benchmark Suite
Kernel: Motif Counting
Author: Xuhao Chen

Will count the number of motifs

Requires input graph:
  - no duplicate edges
  - neighborhoods are sorted by vertex identifiers

motif_base: one thread per vertex using CUDA
*/
#define MAX_SIZE 5
void MotifSolver(Graph &g, unsigned k, std::vector<uint64_t> &count);
//void MotifVerifier(Graph &g, unsigned k, std::vector<uint64_t> count);
