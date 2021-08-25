// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

//#include "common.h"
#include "graph.h"
// k-Cliques Listing (k-CL): list/count the number of cliques of size k 
// Requires input graph:
//  - no self loops
//  - no duplicate edges
//  - neighborhoods are sorted by vertex identifiers

// This implementation reduces the search space by counting 
// each clique only once. This implementation counts cliques 
// in a directed acyclic graph (DAG).

#define MAX_SIZE 5
void KclSolver(Graph &g, unsigned k, uint64_t &total);

