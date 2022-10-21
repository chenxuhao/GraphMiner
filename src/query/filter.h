#include "graph.h"
#include "types.h"
#include "pattern.hh"

class Filter {
private:
  std::string filter_type;

  // Filter schemes: GQL, CFL, DPiso
  static bool GQLFilter(Graph *dg, Pattern *qg, VertexLists &cands, VertexList &cand_count);
  static bool CFLFilter(Graph *dg, Pattern *qg, VertexLists &cands, VertexList &cand_count, VertexList &order, CST &tree);
  static void DPisoFilter(Graph *dg, Pattern *qg, VertexLists &cands, VertexList &cand_count, VertexList &order, CST &tree);
  static bool NLFFilter(const Graph *dg, const Pattern *qg, VertexLists &cands, VertexList &cand_count);

  // Generate filtering plan
  static int generateCFLFilterPlan(const Graph *dg, const Pattern *qg, CST &tree, VertexList &order, VertexList &level_offset);

  // Select start vertex
  static vidType selectTSOFilterStartVertex(const Graph *dg, const Pattern *qg);
  static vidType selectCFLFilterStartVertex(const Graph *dg, const Pattern *qg);
  static vidType selectDPisoStartVertex(const Graph *dg, const Pattern *qg);
  static vidType selectCECIStartVertex(const Graph *dg, const Pattern *qg);

  // Graph operations
  static void bfsTraversal(const Pattern *g, vidType root, CST &tree, VertexList &bfs_order);
  static void dfsTraversal(CST &tree, vidType root, int node_num, VertexList &dfs_order);
  static void dfs(CST &tree, vidType cur_vertex, VertexList &dfs_order, int &count);
  static void old_cheap(int* col_ptrs, int* col_ids, int* match, int* row_match, int n, int m);
  static void match_bfs(int* col_ptrs, int* col_ids, int* match, int* row_match, 
                        int* visited, int* queue, int* previous, int n, int m);

  // Filter operations
  static void sortCandidates(VertexLists &cands, VertexList &cand_count, vidType num);
  static void compactCandidates(VertexLists &cands, VertexList &cand_count, vidType qv_num);
  static bool isCandidateSetValid(VertexLists &cands, VertexList &cand_count, vidType qv_num);
  static void computeCandidateWithNLF(const Graph *dg, const Pattern *qg, vidType qv, vidType &count, VertexList &buffer);
  static bool verifyExactTwigIso(const Graph *dg, const Pattern *qg, vidType dv, vidType qv,
                                 std::vector<std::vector<bool>> &valid_candidates, int *left_to_right_offset, 
                                 int *left_to_right_edges, int *left_to_right_match, int *right_to_left_match, 
                                 int* match_visited, int* match_queue, int* match_previous);
  static void generateCandidates(const Graph *dg, const Pattern *qg, vidType query_vertex,
      vidType *pivot_vertices, int pivot_vertices_count, VertexLists &candidates,
      VertexList &candidates_count, VertexList &flag, VertexList &updated_flag);
  static void pruneCandidates(const Graph *dg, const Pattern *qg, vidType query_vertex,
      vidType *pivot_vertices, int pivot_vertices_count, VertexLists &candidates,
      VertexList &candidates_count, VertexList &flag, VertexList &updated_flag);
public:
  Filter(std::string t) : filter_type(t) {}
  void filtering(Graph *dg, Pattern *qg, VertexLists &cands, VertexList &cand_count);
  static void buildTables(Graph *dg, Pattern *qg, VertexLists &cands, VertexList &cand_count, Edges ***edge_matrix);
  static void printTableCardinality(const Pattern *query_graph, Edges ***edge_matrix);
  static size_t computeMemoryCostInBytes(const Pattern* query_graph, VertexList &candidates_count, Edges ***edge_matrix);
  void printCandidatesInfo(const Pattern *qg, VertexList &candidates_count, std::vector<int> &optimal_candidates_count);
  double computeCandidatesFalsePositiveRatio(const Graph *dg, const Pattern *qg, VertexLists &candidates,
                                             VertexList &candidates_count, std::vector<int> &optimal_candidates_count);
};
