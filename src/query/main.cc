// Copyright 2021 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "pattern.hh"

void QuerySolver(Graph &g, Pattern &p, uint64_t &total, bool use_filter, int n_devices, int chunk_size);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <filename> \n";
    std::cout << "Example: ./query_omp_base ../inputs/citeseer/graph ../inputs/q2-triangle.graph\n";
    exit(1);
  }
  bool use_filter = 1;
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 3) use_filter = atoi(argv[3]);
  if (argc > 4) n_devices = atoi(argv[4]);
  if (argc > 5) chunk_size = atoi(argv[5]);

  std::cout << "Labeled Graph Querying\n";
  Pattern p(argv[2]);
  std::cout << "Pattern: " << p.get_name() << " " << p << "\n";
  p.print_meta_data();
  p.print_graph();
  std::cout << long_separator;

  Graph g(argv[1], 0, 0, 1, 0);
  std::cout << "Data Graph Meta Information\n";
  g.print_meta_data();
  g.BuildReverseIndex();
  std::cout << long_separator;

  uint64_t num = 0;
  QuerySolver(g, p, num, use_filter, n_devices, chunk_size);
  std::cout << "Number of matches: " << num << "\n";
  return 0;
}

