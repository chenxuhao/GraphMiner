// Copyright 2021 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#pragma once
#include "graph.h"
#include "pattern.hh"
// Labeled Graph Querying
// list the occurrances of labeled patterns 
void QuerySolver(Graph &g, Pattern &p, uint64_t &total, int n_devices, int chunk_size);
