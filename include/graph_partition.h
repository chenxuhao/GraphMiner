// Graph partitioner

#pragma once

#include "graph.h"

typedef std::pair<vidType, vidType> Edge;
typedef std::vector<Edge> EdgeList;

class PartitionedGraph {
private:
  Graph *g;
  int num_vertex_chunks;       // number of clusters, i.e., vertex subsets or segments
  int num_subgraphs;           // number of subgraphs, i.e., number of partitions
  int num_2D_partitions;       // number of partitions, i.e., square of 'num_vertex_chunks'
  std::string partitioned_file_path;     // file path for partitioned graph

  VertexLists verts_of_clusters;           // vertices in each cluster 
  std::vector<int> vertex_rank_in_cluster; // the rank of the vertex within its cluster
  std::vector<Graph*> subgraphs;
  std::vector<vidType> begin_vids, end_vids, local_begin, local_end;
  VertexLists idx_map;                      // local to global vertex id mapping

  // generate the vertex-induced subgraph from a vertex subset 
  void generate_induced_subgraph(std::vector<int8_t> v_masks, Graph *g, Graph *subg, int i);

public:
  // constructors
  PartitionedGraph() : num_vertex_chunks(0), num_2D_partitions(0) {}
  PartitionedGraph(Graph *graph, int nc) : g(graph), num_vertex_chunks(nc) { assert(nc>1); }
  PartitionedGraph(Graph *graph, int nc, std::vector<int> cluster_ids);

  Graph* get_subgraph(int i) { return subgraphs[i]; }
  int get_num_subgraphs() { return num_subgraphs; }
  vidType get_local_begin(int i) { return local_begin[i]; } // get local id of the first master vertex for the i-th subgraph
  vidType get_local_end(int i) { return local_end[i]; } // get local id of the last master vertex for the i-th subgraph

  // naive 1D partitioning, i.e., edge-cut
  void edgecut_partition1D();

  // edge-cut 1D partitioning; generate a vertex-induced subgraph for each partition
  void edgecut_induced_partition1D();

  // CSR segmenting
  // Yunming Zhang et. al., Making caches work for graph analytics,
  // 2017 IEEE International Conference on Big Data (Big Data),
  // https://ieeexplore.ieee.org/document/8257937
  void csr_segmenting();

  // naive 2D partitioning
  // partition the graph g according to the cluster id of each vertex
  void partition2D(std::vector<int> cluster_ids);

  // given the ids of clusters, fetch the edges between vertices in these clusters, and form a subgraph in CSR format
  void fetch_partitions(std::string path, std::vector<int> clusters, Graph *& subg);

  void print_subgraphs();
};

