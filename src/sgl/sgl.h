// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#pragma once
#include "graph.h"
#include "pattern.h"
// Subgraph Listing/Counting (SL/SC)
// list/count the occurrances of a given arbitrary pattern

void SglSolver(Graph &g, Pattern &p, uint64_t &total, int n_devices, int chunk_size);
