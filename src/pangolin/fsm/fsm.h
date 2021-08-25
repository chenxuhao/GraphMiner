// Copyright 2021 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
// Frequent Subgraph Mining (FSM): find frequent patterns in an labelled undirected graph 

// Requires input graph:
//  - no self loops
//  - no duplicate edges
//  - neighborhoods are sorted by vertex identifiers

#define MAX_SIZE 5
#define MAX_NUM_PATTERNS 21251
void FsmSolver(Graph &g, unsigned k, unsigned minsup, int nlabels, int &total_num);
