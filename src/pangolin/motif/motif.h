// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
// Motif Counting: count the number of motifs
// Requires input graph:
//  - no self loops
//  - no duplicate edges
//  - neighborhoods are sorted by vertex identifiers

#define MAX_SIZE 5
void MotifSolver(Graph &g, unsigned k, std::vector<uint64_t> &count);
