#pragma once
#include "graph.h"
#include "types.h"
#include "pattern.hh"

class QueryPlan {
private:
  std::string order_type;
  std::string explore_type;
  void generateGQLQueryPlan(const Graph *dg, const Pattern *qg, VertexList &candidates_count, VertexList &order, VertexList &pivot);
  void checkQueryPlanCorrectness(const Pattern *qg, const VertexList &order, const VertexList &pivot) const;
  vidType selectGQLStartVertex(const Pattern* qg, VertexList &candidates_count);
  vidType selectParallelStartVertex(const Pattern* qg, VertexList &candidates_count);
  void updateValidVertices(const Pattern* qg, vidType qv, std::vector<bool> &visited, std::vector<bool> &adjacent);
  void generateBN(const Pattern* query_graph, const VertexList &order, VertexLists &bn, VertexList &bn_count);
  void generateValidCandidateIndex(const Graph *dg, vidType depth, VertexList &embedding, VertexList &idx_embedding,
                                   std::vector<bool> &visited_vertices, VertexLists &bn, VertexList &bn_cnt,
                                   VertexList &order, VertexList &pivot, VertexLists &candidates, Edges ***edge_matrix,
                                   VertexList &idx_count, VertexLists &valid_candidate_index);
  void generateValidCandidateIndex(int depth, VertexList &idx_embedding, VertexLists &bn, VertexList &bn_cnt,
                                   Edges ***edge_matrix, VertexList &order, int *&temp_buffer,
                                   VertexList &idx_count, int *&valid_candidate_index);
  void generateValidCandidates(const Graph *dg, int depth, VertexList &embedding,
                               std::vector<bool> &visited_vertices,
                               VertexLists &bn, VertexList &bn_cnt, VertexList &order,
                               VertexLists &candidates, VertexList &candidates_count,
                               VertexList &idx_count, VertexLists &valid_candidate);
  // exploration methods
  uint64_t exploreGraph(const Graph *dg, const Pattern *qg, VertexLists &candidates, VertexList &candidates_count,
                       VertexList &order, VertexList &pivot, Edges ***edge_matrix, size_t output_limit_num, size_t &call_count);
  uint64_t exploreGQL(const Graph *dg, const Pattern *qg, VertexLists &candidates, VertexList &candidates_count, 
                      VertexList &order, size_t output_limit_num, size_t &call_count);
  uint64_t exploreGQLparallel(const Graph *dg, const Pattern *qg, VertexLists &candidates, VertexList &candidates_count, 
                              VertexList &order, size_t &call_count);
  uint64_t exploreLFTJ(const Graph *dg, const Pattern *qg, VertexLists &candidates, VertexList &candidates_count,
                       VertexList &order, Edges ***edge_matrix, size_t output_limit_num, size_t &call_count);
  uint64_t exploreLFTJparallel(const Graph *dg, const Pattern *qg, VertexLists &candidates, VertexList &candidates_count,
                               VertexList &order, Edges ***edge_matrix, size_t &call_count);
public:
  QueryPlan(std::string otype, std::string etype) : order_type(otype), explore_type(etype) {}
  void generate(const Graph *dg, const Pattern *qg, VertexList &candidates_count, VertexList &order, VertexList &pivot);
  void printSimplifiedQueryPlan(const Pattern *query_graph, const VertexList &order) const;
  void printQueryPlan(const Pattern *query_graph, const VertexList &order) const;
  uint64_t explore(const Graph *dg, const Pattern *qg, VertexLists &candidates, VertexList &candidates_count, 
                   VertexList &order, VertexList &pivot, Edges ***edge_matrix, size_t &call_count);
};

