// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "string.h"

void DiamondSolver(Graph &g, int k, uint64_t &total, int, int);
void CliqueSolver(Graph &g, int k, uint64_t &total, int, int);

int main(int argc, char *argv[]) {
	if (argc < 4) {
		cout << "number of args: " << argc << ", expected 4\n";
		cout << "don't forget fourth argument, diamond or clique\n"; 
		std::cout << "Usage: " << argv[0] << "<graph> <k> [ngpu(0)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph 4\n";
 	  exit(1);
  }
	if (strcmp(argv[3], "diamond")!=0 && strcmp(argv[3],"clique")!=0){
		cout << "last arg must be diamond or clique\n";
		exit(1);
	}
	if (strcmp(argv[3],"diamond")==0){
		cout << "diamond listing with undirected graphs\n";
	} else {
		cout << "k-clique listing with undirected graphs\n";
	}
	if (USE_DAG) std::cout << "Using DAG (static orientation)\n";
  // Graph g(argv[1], USE_DAG); // use DAG
	Graph g(argv[1], false);
	
	int k = atoi(argv[2]);
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 3) n_devices = atoi(argv[3]);
  if (argc > 4) chunk_size = atoi(argv[4]);
  g.print_meta_data();
	
	if (strcmp(argv[3],"diamond")==0){
  	uint64_t total_diamonds = 0;
  	DiamondSolver(g, k, total_diamonds, n_devices, chunk_size);
		std::cout << "num_diamonds = " << total_diamonds << "\n";		
	} else {
		uint64_t total_kcliques = 0;
		CliqueSolver(g, k, total_kcliques, n_devices, chunk_size);	
  	std::cout << "num_" << k << "-cliques = " << total_kcliques << "\n";
	}
  return 0;
}

