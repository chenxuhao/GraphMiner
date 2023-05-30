// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"


int main(int argc, char *argv[]) {
  srand ( time(NULL) );
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)] [chunk_size(1024)] [adj_sorted(1)] [subgraph_profile(0)] [avg_degree_threshold(0)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "Approximate Counting: assuming the neighbor lists are sorted.\n";
  Graph g(argv[1], USE_DAG); // use DAG
  //g.print_graph();
  int n_devices = 1;
  int chunk_size = 1024;
  int adj_sorted = 1;
  int subgraph_profile = 0;
    int threshold = 0;



  if (argc > 2) n_devices = atoi(argv[2]);
  if (argc > 3) chunk_size = atoi(argv[3]);
  g.print_meta_data();
  if (argc > 4) adj_sorted = atoi(argv[4]);
  if (!adj_sorted) g.sort_neighbors();
  if (argc > 5) subgraph_profile = atoi(argv[5]);
  if (argc > 6) threshold = atoi(argv[6]);


  if(subgraph_profile) {
      g.color_sparsify_fast(threshold);
      g.sample_tree_subgraph(threshold);
  } else {
      g.color_sparsify(threshold);
      g.sample_tree(threshold);
  }

  return 0;
}

