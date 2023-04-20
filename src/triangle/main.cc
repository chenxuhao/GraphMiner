// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void TCSolver(Graph &g, uint64_t &total, int n_gpu, int chunk_size, int threshold);

int main(int argc, char *argv[]) {
  if (argc < 4) {
		cout << "third argument is dag (1) or non-dag (0)" << '\n';
    cout << "last argument is degree threshold for high vertices (this value is ignored in omp_base)" << '\n';
    cout << "example: ../../bin/tc_omp_a0 ../../inputs/graph/mico 1 350" << '\n'; 
    std::cout << "Usage (old): " << argv[0] << " <graph> [num_gpu(1)] [chunk_size(1024)] [adj_sorted(1)]\n";
    std::cout << "Example (old): " << argv[0] << " /graph_inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "Triangle Counting: we assume the neighbor lists are sorted.\n";
  // Graph g(argv[1], USE_DAG); // use DAG
	Graph g(argv[1], atoi(argv[2]));
	int n_devices = 1;
  int chunk_size = 1024;
  int adj_sorted = 1;
  if (argc > 2) n_devices = atoi(argv[2]);
  if (argc > 3) chunk_size = atoi(argv[3]);
  g.print_meta_data();
  if (argc > 4) adj_sorted = atoi(argv[4]);
  if (!adj_sorted) g.sort_neighbors();
  uint64_t total = 0;
  TCSolver(g, total, n_devices, chunk_size, atoi(argv[3]));
  std::cout << "total_num_triangles = " << total << "\n";
	return 0;
}

