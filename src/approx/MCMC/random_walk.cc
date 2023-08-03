// Copyright 2023, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void RandomWalk(Graph &g, int k, int64_t &total);

int main(int argc, char *argv[]) {
  srand ( time(NULL) );
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> \n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "Approximate Motif Counting using Random Walk\n";
  Graph g(argv[1]); 
  g.print_meta_data();
  int64_t total = 0;
  RandomWalk(g, k, total);
  std::cout << "total_num_pattern = " << total << "\n";
  return 0;
}

void step(Graph &g) {
}

void RandomWalk(Graph &g, int k, int64_t &total) {
  int num_steps = 100, i = 0;
  while (i < num_steps) {
    step();
    i++;
  }
  if (i < num_steps)
    std::cout << "Stopped at step " << i << " since no adjacent graphlet exists\n";
}
