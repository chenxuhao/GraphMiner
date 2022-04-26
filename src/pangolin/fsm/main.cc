// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "fsm.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << "<graph> [max_size(3)] [min_support(5000)]\n";
    std::cout << "Example: ./bin/pangolin/" << argv[0] << " ./inputs/citeseer/graph 3 5000\n";
    exit(1);
  } 
  unsigned k = 2;
  if (argc > 2) k = atoi(argv[2]);
  unsigned minsup = 5000;
  if (argc > 3) minsup = atoi(argv[3]);
  std::cout << "max_size = " << k << "\n";
  std::cout << "min_support = " << minsup << "\n";

  Graph g(argv[1], 0, 1);
  g.computeLabelsFrequency();
  int nlabels = g.get_vertex_classes();
  g.print_meta_data();
  int num_freqent_patterns = 0;
  FsmSolver(g, k, minsup, nlabels, num_freqent_patterns);
  std::cout << "Number of frequent patterns: " << num_freqent_patterns << "\n";
  return 0;
}

