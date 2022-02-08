#pragma once
#include "graph.h"

class Scheduler {
public:
  Scheduler() : nnz(0) {}
  ~Scheduler() {}
  std::vector<eidType> vertex_chunking(int n, Graph &g, std::vector<vidType*> &src_ptrs, std::vector<vidType*> &dst_ptrs, int stride);
  std::vector<eidType> least_first(int n, Graph &g, std::vector<vidType*> &src_ptrs, std::vector<vidType*> &dst_ptrs, int stride);
  std::vector<eidType> round_robin(int n, Graph &g, std::vector<vidType*> &src_ptrs, std::vector<vidType*> &dst_ptrs, int stride);
private:
  VertexLists srcs;
  VertexLists dsts;
  eidType nnz; // number of tasks/edges
  vidType *src_list, *dst_list; // for COO format
  inline int64_t hop2_workload(Graph &g, vidType src, vidType dst);
  inline int64_t workload_estimate(Graph &g, vidType src, vidType dst);
  inline int smallest_score_id(int n, int64_t* scores);
  eidType init_edgelist(bool sym_break = false, bool ascend = false);
};
