#include <queue>
#include "filter.h"
#define INVALID_VERTEX_ID vidType(-1)
//#define INVALID_VERTEX_ID 100000000
#define OPTIMIZED_LABELED_GRAPH 1

void Filter::filtering(Graph *dg, Pattern *qg, VertexLists &candidates, VertexList &candidates_count) {
  VertexList cfl_order(qg->size()), dpiso_order(qg->size());
  CST cfl_tree, dpiso_tree;
  //std::vector<std::unordered_map<vidType, std::vector<vidType>>> TE_Candidates;
  //std::vector<std::vector<std::unordered_map<vidType, std::vector<vidType>>>> NTE_Candidates;
  if (filter_type == "GQL") {
    //std::cout << "Using GQL filter\n";
    GQLFilter(dg, qg, candidates, candidates_count);
  } else if (filter_type == "CFL") {
    CFLFilter(dg, qg, candidates, candidates_count, cfl_order, cfl_tree);
  } else if (filter_type == "DPiso") {
    DPisoFilter(dg, qg, candidates, candidates_count, dpiso_order, dpiso_tree);
  } else {
    std::cout << "The specified filter type '" << filter_type << "' is not supported." << std::endl;
    exit(-1);
  }
  if (filter_type != "CECI")
    sortCandidates(candidates, candidates_count, qg->size());
}

void Filter::sortCandidates(VertexLists &candidates, VertexList &candidates_count, vidType num) {
  for (vidType i = 0; i < num; ++i)
    std::sort(candidates[i].begin(), candidates[i].begin() + candidates_count[i]);
}

void Filter::buildTables(Graph *dg, Pattern *qg, VertexLists &candidates, VertexList &candidates_count, Edges ***edge_matrix) {
  VertexList flag(dg->size(), 0);
  VertexList updated_flag(dg->size());
  for (vidType i = 0; i < qg->size(); ++i) {
    for (vidType j = 0; j < qg->size(); ++j) {
      edge_matrix[i][j] = NULL;
    }
  }
  std::vector<vidType> build_table_order(qg->size());
  for (vidType i = 0; i < qg->size(); ++i)
    build_table_order[i] = i;
  std::sort(build_table_order.begin(), build_table_order.end(), [qg](vidType l, vidType r) {
    if (qg->get_degree(l) == qg->get_degree(r))
      return l < r;
    return qg->get_degree(l) > qg->get_degree(r);
  });
  std::vector<vidType> temp_edges(dg->sizeEdges());
  for (auto u : build_table_order) {
    vidType u_nbrs_count = qg->get_degree(u);
    vidType updated_flag_count = 0;
    for (vidType i = 0; i < u_nbrs_count; ++i) {
      vidType u_nbr = qg->get_neighbor(u, i);
      if (edge_matrix[u][u_nbr] != NULL)
        continue;
      if (updated_flag_count == 0) {
        for (vidType j = 0; j < candidates_count[u]; ++j) {
          vidType v = candidates[u][j];
          flag[v] = j + 1;
          updated_flag[updated_flag_count++] = v;
        }
      }
      edge_matrix[u_nbr][u] = new Edges;
      edge_matrix[u_nbr][u]->vertex_count_ = candidates_count[u_nbr];
      edge_matrix[u_nbr][u]->offset_ = new vidType[candidates_count[u_nbr] + 1];
      edge_matrix[u][u_nbr] = new Edges;
      edge_matrix[u][u_nbr]->vertex_count_ = candidates_count[u];
      edge_matrix[u][u_nbr]->offset_ = new vidType[candidates_count[u] + 1];
      std::fill(edge_matrix[u][u_nbr]->offset_, edge_matrix[u][u_nbr]->offset_ + candidates_count[u] + 1, 0);
      vidType local_edge_count = 0;
      vidType local_max_degree = 0;
      for (vidType j = 0; j < candidates_count[u_nbr]; ++j) {
        auto v = candidates[u_nbr][j];
        edge_matrix[u_nbr][u]->offset_[j] = local_edge_count;
        auto v_nbrs_count = dg->get_degree(v);
        const vidType* v_nbrs = dg->adj_ptr(v);
        vidType local_degree = 0;
        for (vidType k = 0; k < v_nbrs_count; ++k) {
          vidType v_nbr = v_nbrs[k];
          if (flag[v_nbr] != 0) {
            vidType position = flag[v_nbr] - 1;
            temp_edges[local_edge_count++] = position;
            edge_matrix[u][u_nbr]->offset_[position + 1] += 1;
            local_degree += 1;
          }
        }
        if (local_degree > local_max_degree) {
          local_max_degree = local_degree;
        }
      }
      edge_matrix[u_nbr][u]->offset_[candidates_count[u_nbr]] = local_edge_count;
      edge_matrix[u_nbr][u]->max_degree_ = local_max_degree;
      edge_matrix[u_nbr][u]->edge_count_ = local_edge_count;
      edge_matrix[u_nbr][u]->edge_ = new vidType[local_edge_count];
      std::copy(temp_edges.begin(), temp_edges.begin() + local_edge_count, edge_matrix[u_nbr][u]->edge_);
      edge_matrix[u][u_nbr]->edge_count_ = local_edge_count;
      edge_matrix[u][u_nbr]->edge_ = new vidType[local_edge_count];
      local_max_degree = 0;
      for (vidType j = 1; j <= candidates_count[u]; ++j) {
        if (edge_matrix[u][u_nbr]->offset_[j] > local_max_degree) {
          local_max_degree = edge_matrix[u][u_nbr]->offset_[j];
        }
        edge_matrix[u][u_nbr]->offset_[j] += edge_matrix[u][u_nbr]->offset_[j - 1];
      }
      edge_matrix[u][u_nbr]->max_degree_ = local_max_degree;
      for (vidType j = 0; j < candidates_count[u_nbr]; ++j) {
        vidType begin = j;
        for (vidType k = edge_matrix[u_nbr][u]->offset_[begin]; k < edge_matrix[u_nbr][u]->offset_[begin + 1]; ++k) {
          vidType end = edge_matrix[u_nbr][u]->edge_[k];
          edge_matrix[u][u_nbr]->edge_[edge_matrix[u][u_nbr]->offset_[end]++] = begin;
        }
      }
      for (vidType j = candidates_count[u]; j >= 1; --j) {
        edge_matrix[u][u_nbr]->offset_[j] = edge_matrix[u][u_nbr]->offset_[j - 1];
      }
      edge_matrix[u][u_nbr]->offset_[0] = 0;
    }
    for (vidType i = 0; i < updated_flag_count; ++i) {
      auto v = updated_flag[i];
      flag[v] = 0;
    }
  }
#if ENABLE_QFLITER == 1
  qfliter_bsr_graph_ = new BSRGraph**[qg->size()];
  for (vidType i = 0; i < qg->size(); ++i) {
    qfliter_bsr_graph_[i] = new BSRGraph*[qg->size()];
    for (vidType j = 0; j < qg->size(); ++j) {
      qfliter_bsr_graph_[i][j] = new BSRGraph[qg->size()];
      if (edge_matrix[i][j] != NULL) {
        qfliter_bsr_graph_[i][j]->load(edge_matrix[i][j]->vertex_count_,
            edge_matrix[i][j]->offset_, edge_matrix[i][j]->offset_,
            edge_matrix[i][j]->edge_);
      }
    }
  }
#endif
}

bool Filter::GQLFilter(Graph *dg, Pattern *qg, VertexLists &candidates, VertexList &candidates_count) {
  // Local refinement.
  if (!NLFFilter(dg, qg, candidates, candidates_count))
    return false;

  // Allocate buffer.
  std::vector<std::vector<bool>> valid_candidates(qg->size());
  for (vidType i = 0; i < qg->size(); ++i) {
    valid_candidates[i].resize(dg->size());
    std::fill(valid_candidates[i].begin(), valid_candidates[i].end(), 0);
  }
  auto qg_max_degree = qg->get_max_degree();
  auto dg_max_degree = dg->get_max_degree();
  int* left_to_right_offset = new int[qg_max_degree + 1];
  int* left_to_right_edges = new int[qg_max_degree * dg_max_degree];
  int* left_to_right_match = new int[qg_max_degree];
  int* right_to_left_match = new int[dg_max_degree];
  int* match_visited = new int[dg_max_degree + 1];
  int* match_queue = new int[qg->size()];
  int* match_previous = new int[dg_max_degree + 1];

  // Record valid candidate vertices for each query vertex.
  std::cout << "Record valid candidate vertices for each query vertex\n";
  for (vidType i = 0; i < qg->size(); ++i) {
    for (vidType j = 0; j < candidates_count[i]; ++j) {
      auto data_vertex = candidates[i][j];
      valid_candidates[i][data_vertex] = true;
    }
  }
  // Global refinement.
  std::cout << "Global refinement\n";
  for (vidType l = 0; l < 2; ++l) {
    for (vidType i = 0; i < qg->size(); ++i) {
      auto num = candidates_count[i];
      std::cout << "l = " << l << " qv = " << i << " count = " << num << "\n";
      for (vidType j = 0; j < num; ++j) {
        auto data_vertex = candidates[i][j];
        if (data_vertex == INVALID_VERTEX_ID) continue;
        if (!verifyExactTwigIso(dg, qg, data_vertex, i, valid_candidates,
                                left_to_right_offset, left_to_right_edges, left_to_right_match,
                                right_to_left_match, match_visited, match_queue, match_previous)) {
          candidates[i][j] = INVALID_VERTEX_ID;
          valid_candidates[i][data_vertex] = false;
        }
      }
    }
  }
  // Compact candidates.
  std::cout << "Compact candidates\n";
  compactCandidates(candidates, candidates_count, qg->size());
  delete[] left_to_right_offset;
  delete[] left_to_right_edges;
  delete[] left_to_right_match;
  delete[] right_to_left_match;
  delete[] match_visited;
  delete[] match_queue;
  delete[] match_previous;
  std::cout << "isCandidateSetValid\n";
  return isCandidateSetValid(candidates, candidates_count, qg->size());
}

bool Filter::CFLFilter(Graph *dg, Pattern*qg, VertexLists &candidates, VertexList &candidates_count, VertexList &order, CST &tree) {
  VertexList level_offset(qg->size()+1);
  auto level_count = generateCFLFilterPlan(dg, qg, tree, order, level_offset);
  auto w0 = order[0]; // start vertex
  computeCandidateWithNLF(dg, qg, w0, candidates_count[w0], candidates[w0]);
  VertexList updated_flag(dg->size());
  VertexList flag(dg->size(), 0);

  // Top-down generation.
  for (int i = 1; i < level_count; ++i) {
    // Forward generation.
    for (int j = level_offset[i]; j < level_offset[i+1]; ++j) {
      auto qv = order[j];
      auto& node = tree[qv];
      generateCandidates(dg, qg, qv, node.bn_, node.bn_count_, candidates, candidates_count, flag, updated_flag);
    }
    // Backward prune.
    for (int j = level_offset[i+1] - 1; j >= level_offset[i]; --j) {
      auto qv = order[j];
      auto& node = tree[qv];
      if (node.fn_count_ > 0) {
        pruneCandidates(dg, qg, qv, node.fn_, node.fn_count_, candidates, candidates_count, flag, updated_flag);
      }
    }
  }
  // Bottom-up refinement.
  for (int i = level_count - 2; i >= 0; --i) {
    for (int j = level_offset[i]; j < level_offset[i + 1]; ++j) {
      auto query_vertex = order[j];
      auto& node = tree[query_vertex];
      if (node.under_level_count_ > 0) {
        pruneCandidates(dg, qg, query_vertex, node.under_level_, node.under_level_count_, candidates, candidates_count, flag, updated_flag);
      }
    }
  }
  compactCandidates(candidates, candidates_count, qg->size());
  return isCandidateSetValid(candidates, candidates_count, qg->size());
}

void Filter::DPisoFilter(Graph *dg, Pattern *qg, VertexLists &candidates, VertexList &candidates_count, VertexList &order, CST &tree) {
}

bool Filter::NLFFilter(const Graph *dg, const Pattern *qg, VertexLists &candidates, VertexList &candidates_count) {
  //std::cout << "NLFFilter\n";
  for (vidType i = 0; i < qg->size(); ++i) {
    computeCandidateWithNLF(dg, qg, i, candidates_count[i], candidates[i]);
    std::cout << "qv " << i << " candidates_size = " << candidates_count[i] << "\n";
    if (candidates_count[i] == 0) return false;
  }
  return true;
}

void Filter::compactCandidates(VertexLists &candidates, VertexList &candidates_count, vidType nv) {
  //std::cout << "compactCandidates\n";
  for (vidType i = 0; i < nv; ++i) {
    vidType query_vertex = i;
    vidType next_position = 0;
    for (vidType j = 0; j < candidates_count[query_vertex]; ++j) {
      auto data_vertex = candidates[query_vertex][j];
      if (data_vertex != INVALID_VERTEX_ID) {
        candidates[query_vertex][next_position++] = data_vertex;
      }
    }
    candidates_count[query_vertex] = next_position;
  }
}

bool Filter::isCandidateSetValid(VertexLists &candidates, VertexList &candidates_count, vidType nv) {
  for (vidType i = 0; i < nv; ++i) {
    if (candidates_count[i] == 0)
      return false;
  }
  return true;
}

void Filter::computeCandidateWithNLF(const Graph *dg, const Pattern *qg, vidType qv, vidType &count, VertexList &buffer) {
  //std::cout << "computeCandidateWithNLF, qv = " << qv << ", buffer_size = " << buffer.size() << "\n";
  auto label = qg->get_vlabel(qv);
  auto degree = qg->get_degree(qv);
#if OPTIMIZED_LABELED_GRAPH == 1
  const std::unordered_map<vlabel_t, vidType>* query_vertex_nlf = qg->getVertexNLF(qv);
#endif
  vidType data_vertex_num;
  auto data_vertices = dg->getVerticesByLabel(label, data_vertex_num);
  count = 0;
  for (vidType j = 0; j < data_vertex_num; ++j) {
    auto data_vertex = data_vertices[j];
    if (dg->get_degree(data_vertex) >= degree) {
      // NFL check
#if OPTIMIZED_LABELED_GRAPH == 1
      const std::unordered_map<vlabel_t, vidType>* data_vertex_nlf = dg->getVertexNLF(data_vertex);
      if (data_vertex_nlf->size() >= query_vertex_nlf->size()) {
        bool is_valid = true;
        for (auto element : *query_vertex_nlf) {
          auto iter = data_vertex_nlf->find(element.first);
          if (iter == data_vertex_nlf->end() || iter->second < element.second) {
            is_valid = false;
            break;
          }
        }
        if (is_valid) {
          if (!buffer.empty()) buffer[count] = data_vertex;
          count += 1;
        }
      }
#else
      if (!buffer.empty()) buffer[count] = data_vertex;
      count += 1;
#endif
    }
  }
}

bool Filter::verifyExactTwigIso(const Graph *dg, const Pattern *qg, vidType data_vertex, vidType query_vertex,
                                std::vector<std::vector<bool>> &valid_candidates, int *left_to_right_offset, 
                                int *left_to_right_edges, int *left_to_right_match, int *right_to_left_match, 
                                int* match_visited, int* match_queue, int* match_previous) {
  // Construct the bipartite graph between N(query_vertex) and N(data_vertex)
  int left_partition_size = qg->get_degree(query_vertex);
  int right_partition_size = dg->get_degree(data_vertex);
  const vidType* data_vertex_neighbors = dg->adj_ptr(data_vertex);
  vidType edge_count = 0;
  for (int i = 0; i < left_partition_size; ++i) {
    auto q_neighbor = qg->get_neighbor(query_vertex, i);
    left_to_right_offset[i] = edge_count;
    for (int j = 0; j < right_partition_size; ++j) {
      auto d_neighbor = data_vertex_neighbors[j];
      if (valid_candidates[q_neighbor][d_neighbor]) {
        left_to_right_edges[edge_count++] = j;
      }
    }
  }
  left_to_right_offset[left_partition_size] = edge_count;
  memset(left_to_right_match, -1, left_partition_size * sizeof(int));
  memset(right_to_left_match, -1, right_partition_size * sizeof(int));
  match_bfs(left_to_right_offset, left_to_right_edges, left_to_right_match, right_to_left_match,
            match_visited, match_queue, match_previous, left_partition_size, right_partition_size);
  for (int i = 0; i < left_partition_size; ++i) {
    if (left_to_right_match[i] == -1)
      return false;
  }
  return true;
}

void Filter::printTableCardinality(const Pattern *qg, Edges ***edge_matrix) {
  std::vector<std::pair<std::pair<vidType, vidType >, vidType>> core_edges;
  std::vector<std::pair<std::pair<vidType, vidType >, vidType>> tree_edges;
  std::vector<std::pair<std::pair<vidType, vidType >, vidType>> leaf_edges;
  double sum = 0;
  for (vidType i = 0; i < qg->size(); ++i) {
    vidType begin_vertex = i;
    for (vidType j = i + 1; j < qg->size(); ++j) {
      vidType end_vertex = j;
      if (qg->is_connected(begin_vertex, end_vertex)) {
        vidType cardinality = (*edge_matrix[begin_vertex][end_vertex]).edge_count_;
        sum += cardinality;
        if (qg->getCoreValue(begin_vertex) > 1 && qg->getCoreValue(end_vertex) > 1) {
          core_edges.emplace_back(std::make_pair(std::make_pair(begin_vertex, end_vertex), cardinality));
        }
        else if (qg->get_degree(begin_vertex) == 1 || qg->get_degree(end_vertex) == 1) {
          leaf_edges.emplace_back(std::make_pair(std::make_pair(begin_vertex, end_vertex), cardinality));
        }
        else {
          tree_edges.emplace_back(std::make_pair(std::make_pair(begin_vertex, end_vertex), cardinality));
        }
      }
    }
  }
  printf("Index Info: CoreTable(%zu), TreeTable(%zu), LeafTable(%zu)\n", core_edges.size(), tree_edges.size(), leaf_edges.size());
  for (auto table_info : core_edges) {
    printf("CoreTable %u-%u: %u\n", table_info.first.first, table_info.first.second, table_info.second);
  }
  for (auto table_info : tree_edges) {
    printf("TreeTable %u-%u: %u\n", table_info.first.first, table_info.first.second, table_info.second);
  }
  for (auto table_info : leaf_edges) {
    printf("LeafTable %u-%u: %d\n", table_info.first.first, table_info.first.second, table_info.second);
  }
  printf("Total Cardinality: %.1lf\n", sum);
}

size_t Filter::computeMemoryCostInBytes(const Pattern* qg, VertexList &candidates_count, Edges ***edge_matrix) {
  size_t memory_cost_in_bytes = 0;
  size_t per_element_size = sizeof(vidType);
  for (vidType i = 0; i < qg->size(); ++i) {
    memory_cost_in_bytes += candidates_count[i] * per_element_size;
  }
  for (vidType i = 0; i < qg->size(); ++i) {
    vidType begin_vertex = i;
    for (vidType j = 0; j < qg->size(); ++j) {
      vidType end_vertex = j;
      if (begin_vertex < end_vertex && qg->is_connected(begin_vertex, end_vertex)) {
        Edges& edge = *edge_matrix[begin_vertex][end_vertex];
        memory_cost_in_bytes += edge.edge_count_ * per_element_size + edge.vertex_count_ * per_element_size;
        Edges& reverse_edge = *edge_matrix[end_vertex][begin_vertex];
        memory_cost_in_bytes += reverse_edge.edge_count_ * per_element_size + reverse_edge.vertex_count_ * per_element_size;
      }
    }
  }
  return memory_cost_in_bytes;
}
/*
size_t Filter::computeMemoryCostInBytes(const Pattern* qg, VertexList &candidates_count, VertexList &order, CST &tree,
    std::vector<std::unordered_map<vidType, std::vector<vidType >>> &TE_Candidates,
    std::vector<std::vector<std::unordered_map<vidType, std::vector<vidType>>>> &NTE_Candidates) {
  size_t memory_cost_in_bytes = 0;
  size_t per_element_size = sizeof(vidType);
  for (vidType i = 0; i < qg->size(); ++i) {
    memory_cost_in_bytes += candidates_count[i] * per_element_size;
  }
  for (vidType i = 1; i < qg->size(); ++i) {
    vidType u = order[i];
    TreeNode& u_node = tree[u];
    // NTE_Candidates
    for (vidType j = 0; j < u_node.bn_count_; ++j) {
      vidType u_bn = u_node.bn_[j];
      memory_cost_in_bytes += NTE_Candidates[u][u_bn].size() * per_element_size;
      for (auto iter = NTE_Candidates[u][u_bn].begin(); iter != NTE_Candidates[u][u_bn].end(); ++iter) {
        memory_cost_in_bytes += iter->second.size() * per_element_size;
      }
    }
    // TE_Candidates
    memory_cost_in_bytes += TE_Candidates[u].size() * per_element_size;
    for (auto iter = TE_Candidates[u].begin(); iter != TE_Candidates[u].end(); ++iter) {
      memory_cost_in_bytes += iter->second.size() * per_element_size;
    }
  }
  return memory_cost_in_bytes;
}
*/

// generate filtering plan: CFL, DPiso, CECI, TSO
int Filter::generateCFLFilterPlan(const Graph *dg, const Pattern *qg, CST &tree, VertexList &order, VertexList &level_offset) {
  int level_count = 0;
  auto start_vertex = selectCFLFilterStartVertex(dg, qg);
  bfsTraversal(qg, start_vertex, tree, order);
  std::vector<int> order_index(qg->size());
  for (int i = 0; i < qg->size(); ++i) {
    auto query_vertex = order[i];
    order_index[query_vertex] = i;
  }
  level_count = -1;
  for (int i = 0; i < qg->size(); ++i) {
    auto u = order[i];
    tree[u].under_level_count_ = 0;
    tree[u].bn_count_ = 0;
    tree[u].fn_count_ = 0;
    if (tree[u].level_ != level_count) {
      level_count += 1;
      level_offset[level_count] = 0;
    }
    level_offset[level_count] += 1;
    int u_nbrs_count = qg->get_degree(u);
    for (int j = 0; j < u_nbrs_count; ++j) {
      auto u_nbr = qg->get_neighbor(u, j);
      if (tree[u].level_ == tree[u_nbr].level_) {
        if (order_index[u_nbr] < order_index[u])
          tree[u].bn_[tree[u].bn_count_++] = u_nbr;
        else tree[u].fn_[tree[u].fn_count_++] = u_nbr;
      } else if (tree[u].level_ > tree[u_nbr].level_) {
        tree[u].bn_[tree[u].bn_count_++] = u_nbr;
      } else {
        tree[u].under_level_[tree[u].under_level_count_++] = u_nbr;
      }
    }
  }
  level_count += 1;
  int prev_value = 0;
  for (int i = 1; i <= level_count; ++i) {
    auto temp = level_offset[i];
    level_offset[i] = level_offset[i - 1] + prev_value;
    prev_value = temp;
  }
  level_offset[0] = 0;
  return level_count;
}
/*
void Filter::generateTSOFilterPlan(const Graph *dg, const Graph *qg, CST &tree, VertexList &order) {
  auto start_vertex = selectTSOFilterStartVertex(dg, qg);
  VertexList bfs_order(qg->size());
  bfsTraversal(qg, start_vertex, tree, bfs_order);
  dfsTraversal(tree, start_vertex, qg->size(), order);
}

void GenerateFilteringPlan::generateDPisoFilterPlan(const Graph *dg, const Graph *qg, CST &tree, VertexList &order) {
  auto start_vertex = selectDPisoStartVertex(dg, qg);
  bfsTraversal(qg, start_vertex, tree, order);
  std::vector<int> order_index(qg->size());
  for (int i = 0; i < qg->size(); ++i) {
    auto query_vertex = order[i];
    order_index[query_vertex] = i;
  }
  for (int i = 0; i < qg->size(); ++i) {
    auto u = order[i];
    tree[u].under_level_count_ = 0;
    tree[u].bn_count_ = 0;
    tree[u].fn_count_ = 0;
    int u_nbrs_count = qg->get_degree(u);
    for (int j = 0; j < u_nbrs_count; ++j) {
      auto u_nbr = qg->get_neighbor(u, j);
      if (order_index[u_nbr] < order_index[u]) {
        tree[u].bn_[tree[u].bn_count_++] = u_nbr;
      }
      else {
        tree[u].fn_[tree[u].fn_count_++] = u_nbr;
      }
    }
  }
}

void Filter::generateCECIFilterPlan(const Graph *dg, const Graph *qg, CST &tree, VertexList &order) {
  auto start_vertex = selectCECIStartVertex(dg, qg);
  bfsTraversal(qg, start_vertex, tree, order);
  std::vector<int> order_index(qg->size());
  for (int i = 0; i < qg->size(); ++i) {
    auto query_vertex = order[i];
    order_index[query_vertex] = i;
  }
  for (int i = 0; i < qg->size(); ++i) {
    auto u = order[i];
    tree[u].under_level_count_ = 0;
    tree[u].bn_count_ = 0;
    tree[u].fn_count_ = 0;
    int u_nbrs_count = qg->get_degree(u);
    for (int j = 0; j < u_nbrs_count; ++j) {
      auto u_nbr = qg->get_neighbor(u, j);
      if (u_nbr != tree[u].parent_ && order_index[u_nbr] < order_index[u]) {
        tree[u].bn_[tree[u].bn_count_++] = u_nbr;
        tree[u_nbr].fn_[tree[u_nbr].fn_count_++] = u;
      }
    }
  }
}
*/

void Filter::generateCandidates(const Graph *dg, const Pattern *qg, vidType query_vertex,
                                vidType *pivot_vertices, int pivot_vertices_count, VertexLists &candidates,
                                VertexList &candidates_count, VertexList &flag, VertexList &updated_flag) {
  auto q_label = qg->get_vlabel(query_vertex);
  auto query_vertex_degree = qg->get_degree(query_vertex);
#if OPTIMIZED_LABELED_GRAPH == 1
  const std::unordered_map<vlabel_t, int>* query_vertex_nlf = qg->getVertexNLF(query_vertex);
#endif
  int count = 0;
  int updated_flag_count = 0;
  for (int i = 0; i < pivot_vertices_count; ++i) {
    auto pivot_vertex = pivot_vertices[i];
    for (int j = 0; j < candidates_count[pivot_vertex]; ++j) {
      auto v = candidates[pivot_vertex][j];
      if (v == INVALID_VERTEX_ID) continue;
      for (auto v_nbr : dg->N(v)) {
        auto v_nbr_label = dg->get_vlabel(v_nbr);
        auto v_nbr_degree = dg->get_degree(v_nbr);
        if (flag[v_nbr] == count && v_nbr_label == q_label && v_nbr_degree >= query_vertex_degree) {
          flag[v_nbr] += 1;
          if (count == 0) {
            updated_flag[updated_flag_count++] = v_nbr;
          }
        }
      }
    }
    count += 1;
  }
  for (int i = 0; i < updated_flag_count; ++i) {
    vidType v = updated_flag[i];
    if (flag[v] == count) {
      // NLF filter.
#if OPTIMIZED_LABELED_GRAPH == 1
      const std::unordered_map<vlabel_t, int>* data_vertex_nlf = dg->getVertexNLF(v);
      if (data_vertex_nlf->size() >= query_vertex_nlf->size()) {
        bool is_valid = true;
        for (auto element : *query_vertex_nlf) {
          auto iter = data_vertex_nlf->find(element.first);
          if (iter == data_vertex_nlf->end() || iter->second < element.second) {
            is_valid = false;
            break;
          }
        }
        if (is_valid) {
          candidates[query_vertex][candidates_count[query_vertex]++] = v;
        }
      }
#else
      candidates[query_vertex][candidates_count[query_vertex]++] = v;
#endif
    }
  }
  for (int i = 0; i < updated_flag_count; ++i) {
    auto v = updated_flag[i];
    flag[v] = 0;
  }
}

void Filter::pruneCandidates(const Graph *dg, const Pattern *qg, vidType query_vertex,
    vidType *pivot_vertices, int pivot_vertices_count, VertexLists &candidates,
    VertexList &candidates_count, VertexList &flag, VertexList &updated_flag) {
  auto q_label = qg->get_vlabel(query_vertex);
  auto query_vertex_degree = qg->get_degree(query_vertex);
  int count = 0;
  int updated_flag_count = 0;
  for (int i = 0; i < pivot_vertices_count; ++i) {
    auto pivot_vertex = pivot_vertices[i];
    for (int j = 0; j < candidates_count[pivot_vertex]; ++j) {
      auto v = candidates[pivot_vertex][j];
      if (v == INVALID_VERTEX_ID) continue;
      for (auto v_nbr : dg->N(v)) {
        auto v_nbr_label = dg->get_vlabel(v_nbr);
        auto v_nbr_degree = dg->get_degree(v_nbr);
        if (flag[v_nbr] == count && v_nbr_label == q_label && v_nbr_degree >= query_vertex_degree) {
          flag[v_nbr] += 1;
          if (count == 0) {
            updated_flag[updated_flag_count++] = v_nbr;
          }
        }
      }
    }
    count += 1;
  }
  for (int i = 0; i < candidates_count[query_vertex]; ++i) {
    auto v = candidates[query_vertex][i];
    if (v == INVALID_VERTEX_ID) continue;
    if (flag[v] != count) {
      candidates[query_vertex][i] = INVALID_VERTEX_ID;
    }
  }
  for (int i = 0; i < updated_flag_count; ++i) {
    auto v = updated_flag[i];
    flag[v] = 0;
  }
}

void Filter::printCandidatesInfo(const Pattern *qg, VertexList &candidates_count, std::vector<int> &optimal_candidates_count) {
  std::vector<std::pair<vidType, int>> core_vertices;
  std::vector<std::pair<vidType, int>> tree_vertices;
  std::vector<std::pair<vidType, int>> leaf_vertices;
  auto query_vertices_num = qg->size();
  double sum = 0;
  double optimal_sum = 0;
  for (int i = 0; i < query_vertices_num; ++i) {
    auto cur_vertex = i;
    auto count = candidates_count[cur_vertex];
    sum += count;
    optimal_sum += optimal_candidates_count[cur_vertex];
    if (qg->getCoreValue(cur_vertex) > 1) {
      core_vertices.emplace_back(std::make_pair(cur_vertex, count));
    }
    else {
      if (qg->get_degree(cur_vertex) > 1) {
        tree_vertices.emplace_back(std::make_pair(cur_vertex, count));
      }
      else {
        leaf_vertices.emplace_back(std::make_pair(cur_vertex, count));
      }
    }
  }
  printf("#Candidate Information: CoreVertex(%zu), TreeVertex(%zu), LeafVertex(%zu)\n", core_vertices.size(), tree_vertices.size(), leaf_vertices.size());
  for (auto candidate_info : core_vertices) {
    printf("CoreVertex %u: %u, %u \n", candidate_info.first, candidate_info.second, optimal_candidates_count[candidate_info.first]);
  }
  for (auto candidate_info : tree_vertices) {
    printf("TreeVertex %u: %u, %u\n", candidate_info.first, candidate_info.second, optimal_candidates_count[candidate_info.first]);
  }
  for (auto candidate_info : leaf_vertices) {
    printf("LeafVertex %u: %u, %u\n", candidate_info.first, candidate_info.second, optimal_candidates_count[candidate_info.first]);
  }
  printf("Total #Candidates: %.1lf, %.1lf\n", sum, optimal_sum);
}

double Filter::computeCandidatesFalsePositiveRatio(const Graph *dg, const Pattern *qg, VertexLists &candidates,
    VertexList &candidates_count, std::vector<int> &optimal_candidates_count) {
  auto query_vertices_count = qg->size();
  auto data_vertices_count = dg->size();
  std::vector<std::vector<int>> candidates_copy(query_vertices_count);
  for (int i = 0; i < query_vertices_count; ++i) {
    candidates_copy[i].resize(candidates_count[i]);
    std::copy(candidates[i].begin(), candidates[i].begin() + candidates_count[i], candidates_copy[i].begin());
  }
  std::vector<int> flag(data_vertices_count, 0);
  std::vector<int> updated_flag;
  std::vector<double> per_query_vertex_false_positive_ratio(query_vertices_count);
  optimal_candidates_count.resize(query_vertices_count);
  bool is_steady = false;
  while (!is_steady) {
    is_steady = true;
    for (int i = 0; i < query_vertices_count; ++i) {
      auto u = i;
      int u_nbr_cnt = qg->get_degree(u);
      int valid_flag = 0;
      for (int j = 0; j < u_nbr_cnt; ++j) {
        auto u_nbr = qg->get_neighbor(u, j);
        for (int k = 0; k < candidates_count[u_nbr]; ++k) {
          auto v = candidates_copy[u_nbr][k];
          if (v == INVALID_VERTEX_ID) continue;
          for (auto v_nbr: dg->N(v)) {
            if (flag[v_nbr] == valid_flag) {
              flag[v_nbr] += 1;
              if (valid_flag == 0) {
                updated_flag.push_back(v_nbr);
              }
            }
          }
        }
        valid_flag += 1;
      }
      for (int j = 0; j < candidates_count[u]; ++j) {
        auto v = candidates_copy[u][j];
        if (v == INVALID_VERTEX_ID) continue;
        if (flag[v] != valid_flag) {
          candidates_copy[u][j] = INVALID_VERTEX_ID;
          is_steady = false;
        }
      }
      for (auto v : updated_flag) flag[v] = 0;
      updated_flag.clear();
    }
  }
  double sum = 0;
  for (int i = 0; i < query_vertices_count; ++i) {
    auto u = i;
    int negative_count = 0;
    for (int j = 0; j < candidates_count[u]; ++j) {
      auto v = candidates_copy[u][j];
      if (v == INVALID_VERTEX_ID)
        negative_count += 1;
    }
    per_query_vertex_false_positive_ratio[u] =
      (negative_count) / (double) candidates_count[u];
    sum += per_query_vertex_false_positive_ratio[u];
    optimal_candidates_count[u] = candidates_count[u] - negative_count;
  }
  return sum / query_vertices_count;
}

// Select Start Vertex
vidType Filter::selectTSOFilterStartVertex(const Graph *dg, const Pattern *qg) {
  auto rank_compare = [](std::pair<vidType, double> l, std::pair<vidType, double> r) {
    return l.second < r.second;
  };
  // Maximum priority queue.
  std::priority_queue<std::pair<vidType, double>, std::vector<std::pair<vidType, double>>, decltype(rank_compare)> rank_queue(rank_compare);
  // Compute the ranking.
  for (int i = 0; i < qg->size(); ++i) {
    vidType query_vertex = i;
    auto label = qg->get_vlabel(query_vertex);
    auto degree = qg->get_degree(query_vertex);
    auto frequency = dg->getLabelsFrequency(label);
    double rank = frequency / (double)degree;
    rank_queue.push(std::make_pair(query_vertex, rank));
  }
  // Keep the top-3.
  while (rank_queue.size() > 3) {
    rank_queue.pop();
  }
  // Pick the one with the smallest number of candidates.
  vidType start_vertex = 0;
  auto min_candidates_num = dg->get_max_label_frequency() + 1;
  VertexList empty_buffer;
  while (!rank_queue.empty()) {
    auto query_vertex = rank_queue.top().first;
    if (rank_queue.size() == 1) {
      int count;
      computeCandidateWithNLF(dg, qg, query_vertex, count, empty_buffer);
      if (count < min_candidates_num) {
        start_vertex = query_vertex;
      }
    }
    else {
      auto label = qg->get_vlabel(query_vertex);
      auto frequency = dg->getLabelsFrequency(label);
      if (frequency / (double)dg->size() <= 0.05) {
        int count;
        computeCandidateWithNLF(dg, qg, query_vertex, count, empty_buffer);
        if (count < min_candidates_num) {
          start_vertex = query_vertex;
          min_candidates_num = count;
        }
      }
    }
    rank_queue.pop();
  }
  return start_vertex;
}

vidType Filter::selectCFLFilterStartVertex(const Graph *dg, const Pattern *qg) {
  auto rank_compare = [](std::pair<vidType, double> l, std::pair<vidType, double> r) {
    return l.second < r.second;
  };
  std::priority_queue<std::pair<vidType, double>, std::vector<std::pair<vidType, double>>, decltype(rank_compare)> rank_queue(rank_compare);
  // Compute the ranking.
  for (int i = 0; i < qg->size(); ++i) {
    auto query_vertex = i;
    if (qg->get2CoreSize() == 0 || qg->getCoreValue(query_vertex) > 1) {
      auto label = qg->get_vlabel(query_vertex);
      auto degree = qg->get_degree(query_vertex);
      auto frequency = dg->getLabelsFrequency(label);
      double rank = frequency / (double) degree;
      rank_queue.push(std::make_pair(query_vertex, rank));
    }
  }
  // Keep the top-3.
  while (rank_queue.size() > 3) {
    rank_queue.pop();
  }
  vidType start_vertex = 0;
  double min_score = dg->get_max_label_frequency() + 1;
  VertexList empty_buffer;
  while (!rank_queue.empty()) {
    auto query_vertex = rank_queue.top().first;
    int count;
    computeCandidateWithNLF(dg, qg, query_vertex, count, empty_buffer);
    double cur_score = count / (double) qg->get_degree(query_vertex);
    if (cur_score < min_score) {
      start_vertex = query_vertex;
      min_score = cur_score;
    }
    rank_queue.pop();
  }
  return start_vertex;
}

vidType Filter::selectDPisoStartVertex(const Graph *dg, const Pattern *qg) {
  double min_score = dg->size();
  vidType start_vertex = 0;
  for (int i = 0; i < qg->size(); ++i) {
    auto degree = qg->get_degree(i);
    if (degree <= 1) continue;
    int count = 0;
    //computeCandidateWithLDF(dg, qg, i, count);
    double cur_score = count / (double)degree;
    if (cur_score < min_score) {
      min_score = cur_score;
      start_vertex = i;
    }
  }
  return start_vertex;
}

vidType Filter::selectCECIStartVertex(const Graph *dg, const Pattern *qg) {
  double min_score = dg->size();
  vidType start_vertex = 0;
  VertexList empty_buffer;
  for (int i = 0; i < qg->size(); ++i) {
    auto degree = qg->get_degree(i);
    int count = 0;
    computeCandidateWithNLF(dg, qg, i, count, empty_buffer);
    double cur_score = count / (double)degree;
    if (cur_score < min_score) {
      min_score = cur_score;
      start_vertex = i;
    }
  }
  return start_vertex;
}

// Graph Operations
void Filter::bfsTraversal(const Pattern *graph, vidType root_vertex, CST &tree, VertexList &bfs_order) {
  auto vertex_num = graph->size();
  std::queue<vidType> bfs_queue;
  std::vector<bool> visited(vertex_num, false);
  tree.resize(vertex_num);
  for (int i = 0; i < vertex_num; ++i)
    tree[i].initialize(vertex_num);
  int visited_vertex_count = 0;
  bfs_queue.push(root_vertex);
  visited[root_vertex] = true;
  tree[root_vertex].level_ = 0;
  tree[root_vertex].id_ = root_vertex;
  while(!bfs_queue.empty()) {
    const auto u = bfs_queue.front();
    bfs_queue.pop();
    bfs_order[visited_vertex_count++] = u;
    auto u_nbrs_count = graph->get_degree(u);
    for (int i = 0; i < u_nbrs_count; ++i) {
      auto u_nbr = graph->get_neighbor(u, i);
      if (!visited[u_nbr]) {
        bfs_queue.push(u_nbr);
        visited[u_nbr] = true;
        tree[u_nbr].id_ = u_nbr;
        tree[u_nbr].parent_ = u;
        tree[u_nbr].level_ = tree[u] .level_ + 1;
        tree[u].children_[tree[u].children_count_++] = u_nbr;
      }
    }
  }
}

void Filter::dfsTraversal(CST &tree, vidType root_vertex, int node_num, VertexList &dfs_order) {
  int count = 0;
  dfs(tree, root_vertex, dfs_order, count);
}

void Filter::dfs(CST &tree, vidType cur_vertex, VertexList &dfs_order, int &count) {
  dfs_order[count++] = cur_vertex;
  for (int i = 0; i < tree[cur_vertex].children_count_; ++i) {
    dfs(tree, tree[cur_vertex].children_[i], dfs_order, count);
  }
}

void Filter::old_cheap(int* col_ptrs, int* col_ids, int* match, int* row_match, int n, int m) {
  for (int i = 0; i < n; i++) {
    int s_ptr = col_ptrs[i];
    int e_ptr = col_ptrs[i + 1];
    for (int ptr = s_ptr; ptr < e_ptr; ptr++) {
      int r_id = col_ids[ptr];
      if (row_match[r_id] == -1) {
        match[i] = r_id;
        row_match[r_id] = i;
        break;
      }
    }
  }
}

void Filter::match_bfs(int* col_ptrs, int* col_ids, int* match, int* row_match, 
                       int* visited, int* queue, int* previous, int n, int m) {
  int queue_ptr, queue_col, ptr, next_augment_no, i, j, queue_size, row, col, temp, eptr;
  old_cheap(col_ptrs, col_ids, match, row_match, n, m);
  memset(visited, 0, sizeof(int) * m);
  next_augment_no = 1;
  for (i = 0; i < n; i++) {
    if (match[i] == -1 && col_ptrs[i] != col_ptrs[i+1]) {
      queue[0] = i; queue_ptr = 0; queue_size = 1;
      while (queue_size > queue_ptr) {
        queue_col = queue[queue_ptr++];
        eptr = col_ptrs[queue_col + 1];
        for (ptr = col_ptrs[queue_col]; ptr < eptr; ptr++) {
          row = col_ids[ptr];
          temp = visited[row];
          if (temp != next_augment_no && temp != -1) {
            previous[row] = queue_col;
            visited[row] = next_augment_no;
            col = row_match[row];
            if (col == -1) {
              // Find an augmenting path. Then, trace back and modify the augmenting path.
              while (row != -1) {
                col = previous[row];
                temp = match[col];
                match[col] = row;
                row_match[row] = col;
                row = temp;
              }
              next_augment_no++;
              queue_size = 0;
              break;
            } else {
              // Continue to construct the match.
              queue[queue_size++] = col;
            }
          }
        }
      }
      if (match[i] == -1) {
        for (j = 1; j < queue_size; j++) {
          visited[match[queue[j]]] = -1;
        }
      }
    }
  }
}

