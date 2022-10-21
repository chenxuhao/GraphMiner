#include "query_plan.h"
#include "intersect.h"

// generate a query plan
void QueryPlan::generate(const Graph *dg, const Pattern *qg, VertexList &candidates_count, VertexList &m_order, VertexList &pivot) {
  if (order_type == "GQL") {
    std::cout << "Generating GQL maching order\n";
    generateGQLQueryPlan(dg, qg, candidates_count, m_order, pivot);
  } else {
  }
  if (order_type != "Spectrum") {
    checkQueryPlanCorrectness(qg, m_order, pivot);
    printSimplifiedQueryPlan(qg, m_order);
  }
}

// search space exploration
uint64_t QueryPlan::explore(const Graph *dg, const Pattern *qg, VertexLists &candidates, VertexList &candidates_count, 
                            VertexList &order, VertexList &pivot, Edges ***edge_matrix, size_t &call_count) {
  call_count = 0;
  size_t output_limit = 0;
  output_limit = std::numeric_limits<size_t>::max();
  uint64_t match_count = 0;

  if (explore_type == "EXPLORE") {
    match_count = exploreGraph(dg, qg, candidates, candidates_count, order, pivot, edge_matrix, output_limit, call_count);
  } else if (explore_type == "LFTJ") {
    //match_count = exploreLFTJ(dg, qg, candidates, candidates_count, order, edge_matrix, output_limit, call_count);
    match_count = exploreLFTJparallel(dg, qg, candidates, candidates_count, order, edge_matrix, call_count);
  } else if (explore_type == "GQL") {
    match_count = exploreGQL(dg, qg, candidates, candidates_count, order, output_limit, call_count);
    //match_count = exploreGQLparallel(dg, qg, candidates, candidates_count, order, call_count);
  } else {
    std::cout << "The specified engine type '" << explore_type << "' is not supported." << std::endl;
    exit(-1);
  }
  return match_count;
}

void QueryPlan::generateGQLQueryPlan(const Graph *dg, const Pattern *qg, VertexList &candidates_count, VertexList &order, VertexList &pivot) {
  // Select the vertex v such that (1) v is adjacent to the selected vertices; and (2) v has the minimum number of candidates.
  std::vector<bool> visited_vertices(qg->size(), false);
  std::vector<bool> adjacent_vertices(qg->size(), false);
  //auto start_vertex = selectGQLStartVertex(qg, candidates_count);
  auto start_vertex = selectParallelStartVertex(qg, candidates_count);
  order[0] = start_vertex;
  updateValidVertices(qg, start_vertex, visited_vertices, adjacent_vertices);
  for (vidType i = 1; i < qg->size(); ++i) {
    vidType next_vertex = 0;
    vidType min_value = dg->size() + 1;
    for (vidType j = 0; j < qg->size(); ++j) {
      vidType cur_vertex = j;
      if (!visited_vertices[cur_vertex] && adjacent_vertices[cur_vertex]) {
        if (candidates_count[cur_vertex] < min_value) {
          min_value = candidates_count[cur_vertex];
          next_vertex = cur_vertex;
        }
        else if (candidates_count[cur_vertex] == min_value && qg->get_degree(cur_vertex) > qg->get_degree(next_vertex)) {
          next_vertex = cur_vertex;
        }
      }
    }
    updateValidVertices(qg, next_vertex, visited_vertices, adjacent_vertices);
    order[i] = next_vertex;
  }
  // Pick a pivot randomly.
  for (vidType i = 1; i < qg->size(); ++i) {
    auto u = order[i];
    for (vidType j = 0; j < i; ++j) {
      auto cur_vertex = order[j];
      if (qg->is_connected(u, cur_vertex)) {
        pivot[i] = cur_vertex;
        break;
      }
    }
  }
}

// Select the vertex with the minimum number of candidates as the start vertex.
vidType QueryPlan::selectGQLStartVertex(const Pattern* qg, VertexList &count) {
  // Tie Handling: 1. degree; 2. label id
  vidType start_vertex = 0;
  for (vidType i = 1; i < qg->size(); ++i) {
    auto cur_vertex = i;
    if (count[cur_vertex] < count[start_vertex]) {
      start_vertex = cur_vertex;
    }
    else if (count[cur_vertex] == count[start_vertex]
        && qg->get_degree(cur_vertex) > qg->get_degree(start_vertex)) {
      start_vertex = cur_vertex;
    }
  }
  return start_vertex;
}

// Select the vertex with the minimum number of candidates as the start vertex.
vidType QueryPlan::selectParallelStartVertex(const Pattern* qg, VertexList &count) {
  for (vidType i = 0; i < qg->size(); ++i) {
    std::cout << "count[qv_" << i << "] = " << count[i] << "\n";
  }
  // Tie Handling: 1. degree; 2. label id
  vidType start_vertex = 0;
  for (vidType i = 1; i < qg->size(); ++i) {
    auto cur_vertex = i;
    if (count[cur_vertex] > count[start_vertex])
      start_vertex = cur_vertex;
    else if (count[cur_vertex] == count[start_vertex]
        && qg->get_degree(cur_vertex) > qg->get_degree(start_vertex)) {
      start_vertex = cur_vertex;
    }
  }
  return start_vertex;
}

void QueryPlan::updateValidVertices(const Pattern* qg, vidType qv, std::vector<bool> &visited, std::vector<bool> &adjacent) {
  visited[qv] = true;
  auto nbr_cnt = qg->get_degree(qv);
  for (vidType i = 0; i < nbr_cnt; ++i) {
    auto nbr = qg->get_neighbor(qv, i);
    adjacent[nbr] = true;
  }
}

uint64_t QueryPlan::exploreGraph(const Graph *dg, const Pattern *qg, VertexLists &candidates,
                                 VertexList &candidates_count, VertexList &order, VertexList &pivot, Edges ***edge_matrix,
                                 size_t output_limit_num, size_t &call_count) {
  uint64_t embedding_cnt = 0;

  // Generate the bn.
  int max_depth = qg->size();
  auto start_vertex = order[0];
  VertexLists bn(max_depth);
  for (int i = 0; i < max_depth; ++i)
    bn[i].resize(max_depth);
  VertexList bn_count(max_depth, 0);
  generateBN(qg, order, bn, bn_count);
 
  // Allocate the memory buffer.
  VertexList idx(max_depth, 0);
  VertexList idx_count(max_depth, 0);
  VertexList embedding(max_depth, 0);
  std::vector<bool> visited_vertices(dg->size(), false);
  VertexList idx_embedding(max_depth);
  auto max_candidates_num = candidates_count[0];
  for (vidType i = 0; i < max_depth; ++i) {
    auto cur_count = candidates_count[i];
    if (cur_count > max_candidates_num)
      max_candidates_num = cur_count;
  }
  VertexList temp_buffer(max_candidates_num);
  VertexLists valid_candidate_idx(max_depth);
  for (vidType i = 0; i < max_depth; ++i)
    valid_candidate_idx[i].resize(max_candidates_num);

  // Evaluate the query.
  int cur_depth = 0;
  idx[cur_depth] = 0;
  idx_count[cur_depth] = candidates_count[start_vertex];
  for (vidType i = 0; i < idx_count[cur_depth]; ++i)
    valid_candidate_idx[cur_depth][i] = i;
  while (true) {
    while (idx[cur_depth] < idx_count[cur_depth]) {
      auto valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
      auto u = order[cur_depth];
      auto v = candidates[u][valid_idx];
      embedding[u] = v;
      idx_embedding[u] = valid_idx;
      visited_vertices[v] = true;
      idx[cur_depth] += 1;
      if (cur_depth == max_depth - 1) {
        embedding_cnt += 1;
        visited_vertices[v] = false;
        if (embedding_cnt >= output_limit_num)
          return embedding_cnt;
      } else {
        call_count += 1;
        cur_depth += 1;
        idx[cur_depth] = 0;
        generateValidCandidateIndex(dg, cur_depth, embedding, idx_embedding, visited_vertices, bn, bn_count,
                                    order, pivot, candidates, edge_matrix, idx_count, valid_candidate_idx);
      }
    }
    cur_depth -= 1;
    if (cur_depth < 0) break;
    else visited_vertices[embedding[order[cur_depth]]] = false;
  }
  return embedding_cnt;
}

uint64_t QueryPlan::exploreGQL(const Graph *dg, const Pattern *qg, VertexLists &candidates, 
                               VertexList &candidates_count, VertexList &order, 
                               size_t output_limit_num, size_t &call_count) {
  int max_depth = qg->size();
  auto start_vertex = order[0];
  std::vector<bool> visited_vertices(dg->size(), false);
  VertexLists bn(max_depth);
  for (int j = 0; j < max_depth; ++j)
    bn[j].resize(max_depth);
  VertexList bn_count(max_depth, 0);
  std::vector<bool> visited_query_vertices(max_depth, false);
  visited_query_vertices[start_vertex] = true;
  for (int j = 1; j < max_depth; ++j) {
    auto v = order[j];
    for (auto u : qg->N(v)) {
      if (visited_query_vertices[u])
        bn[j][bn_count[j]++] = u;
    }
    visited_query_vertices[v] = true;
  }
  VertexList idx(max_depth);
  VertexList idx_count(max_depth);
  VertexList embedding(max_depth);
  VertexLists valid_candidate(max_depth);
  for (int j = 0; j < max_depth; ++j) {
    auto nc = candidates_count[order[j]];
    valid_candidate[j].resize(nc);
  }

  std::cout << "Start GQL exploration sequentially\n";
  uint64_t embedding_cnt = 0;
  auto num = candidates_count[start_vertex];
  for (int i = 0; i < num; i++) {
    auto v0 = candidates[start_vertex][i];;
    embedding[start_vertex] = v0;
    visited_vertices[v0] = true; // set mask
    idx[1] = 0;
    int cur_depth = 1;
    generateValidCandidates(dg, 1, embedding, visited_vertices, bn, bn_count, order,
                            candidates, candidates_count, idx_count, valid_candidate);
    while (1) { // Start DFS walk
      while (idx[cur_depth] < idx_count[cur_depth]) {
        auto u = order[cur_depth];
        auto v = valid_candidate[cur_depth][idx[cur_depth]];
        embedding[u] = v;
        visited_vertices[v] = true; // set mask
        idx[cur_depth] += 1;
        if (cur_depth == max_depth - 1) { // arrive the last level; backtrack
          embedding_cnt += 1;
          visited_vertices[v] = false; // resume mask
        } else {
          call_count += 1;
          cur_depth += 1;
          idx[cur_depth] = 0;
          generateValidCandidates(dg, cur_depth, embedding, visited_vertices, bn, bn_count,
                                  order, candidates, candidates_count, idx_count, valid_candidate);
        }
      }
      cur_depth -= 1;
      visited_vertices[embedding[order[cur_depth]]] = false; // resume masks
      if (cur_depth == 0) break;
    }
  }
  return embedding_cnt;
}

uint64_t QueryPlan::exploreGQLparallel(const Graph *dg, const Pattern *qg, VertexLists &candidates, 
                                       VertexList &candidates_count, VertexList &order, size_t &call_count) {
  int max_depth = qg->size();
  auto start_vertex = order[0];
  auto num = candidates_count[start_vertex];
  int nt = 1;
  #pragma omp parallel
  {
    nt = omp_get_num_threads();
  }
  std::vector<std::vector<bool>> masks(nt);
  for (int i = 0; i < nt; i++) {
    masks[i].resize(dg->size());
    std::fill(masks[i].begin(), masks[i].end(), false);
  }
  VertexLists idxes(nt);
  VertexLists bn_counts(nt);
  VertexLists idx_counts(nt);
  VertexLists embeddings(nt);
  VertexLists idx_embeddings(nt);
  std::vector<VertexLists> bns(nt);
  std::vector<VertexLists> valid_candidates(nt);
  for (int i = 0; i < nt; i++) {
    masks[i].resize(dg->size());
    idxes[i].resize(max_depth);
    bns[i].resize(max_depth);
    bn_counts[i].resize(max_depth);
    idx_counts[i].resize(max_depth);
    embeddings[i].resize(max_depth);
    idx_embeddings[i].resize(max_depth);
    valid_candidates[i].resize(max_depth);
    std::fill(masks[i].begin(), masks[i].end(), false);
    std::fill(idxes[i].begin(), idxes[i].end(), 0);
    std::fill(bn_counts[i].begin(), bn_counts[i].end(), 0);
    std::fill(idx_counts[i].begin(), idx_counts[i].end(), 0);
    std::fill(embeddings[i].begin(), embeddings[i].end(), 0);
    for (int j = 1; j < max_depth; ++j) {
      auto nc = candidates_count[order[j]];
      valid_candidates[i][j].resize(nc);
    }
    std::vector<bool> visited_query_vertices(max_depth, false);
    visited_query_vertices[start_vertex] = true;
    for (int j = 1; j < max_depth; ++j) {
      bns[i][j].resize(max_depth);
      auto v = order[j];
      for (auto u : qg->N(v)) {
        if (visited_query_vertices[u])
          bns[i][j][bn_counts[i][j]++] = u;
      }
      visited_query_vertices[v] = true;
    }
  }
 
  std::cout << "Start GQL exploration in parallel\n";
  uint64_t embedding_cnt = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:embedding_cnt,call_count)
  for (int i = 0; i < num; i++) {
    auto tid = omp_get_thread_num();
    auto &visited_vertices = masks[tid];
    auto &idx = idxes[tid];
    auto &bn = bns[tid];
    auto &bn_count = bn_counts[tid];
    auto &idx_count = idx_counts[tid];
    auto &embedding = embeddings[tid];
    auto &valid_candidate = valid_candidates[tid];
  
    auto v0 = candidates[start_vertex][i];;
    embedding[start_vertex] = v0;
    visited_vertices[v0] = true; // set mask
    idx[1] = 0;
    int cur_depth = 1;
    generateValidCandidates(dg, 1, embedding, visited_vertices, bn, bn_count, order,
                            candidates, candidates_count, idx_count, valid_candidate);
    while (1) { // Start DFS walk
      while (idx[cur_depth] < idx_count[cur_depth]) {
        auto u = order[cur_depth];
        auto v = valid_candidate[cur_depth][idx[cur_depth]];
        embedding[u] = v;
        visited_vertices[v] = true; // set mask
        idx[cur_depth] += 1;
        if (cur_depth == max_depth - 1) { // arrive the last level; backtrack
          embedding_cnt += 1;
          visited_vertices[v] = false; // resume mask
        } else {
          call_count += 1;
          cur_depth += 1;
          idx[cur_depth] = 0;
          generateValidCandidates(dg, cur_depth, embedding, visited_vertices, bn, bn_count,
                                  order, candidates, candidates_count, idx_count, valid_candidate);
        }
      }
      cur_depth -= 1;
      visited_vertices[embedding[order[cur_depth]]] = false; // resume masks
      if (cur_depth == 0) break;
    }
  }
  return embedding_cnt;
}

uint64_t QueryPlan::exploreLFTJ(const Graph *dg, const Pattern *qg, VertexLists &candidates, 
                                VertexList &candidates_count, VertexList &order, Edges ***edge_matrix,
                                size_t output_limit_num, size_t &call_count) {
  int max_depth = qg->size();
  auto start_vertex = order[0];
#ifdef DISTRIBUTION
  std::vector<size_t> distribution_count_(dg->size(), 0);
  std::vector<size_t> begin_count(qg->size(), 0);
#endif
  auto max_candidates_num = candidates_count[0];
  for (int j = 1; j < max_depth; ++j) {
    auto cur_count = candidates_count[j];
    if (cur_count > max_candidates_num)
      max_candidates_num = cur_count;
  }
  // Generate bn.
  VertexLists bn(max_depth);
  for (int j = 1; j < max_depth; ++j)
    bn[j].resize(max_depth);
  VertexList bn_count(max_depth, 0);
  generateBN(qg, order, bn, bn_count);

  // Allocate the memory buffer.
  VertexList idx(max_depth, 0);
  VertexList idx_count(max_depth, 0);
  VertexList embedding(max_depth, 0);
  VertexList idx_embedding(max_depth);
  int *temp_buffer = new int[max_candidates_num];
  std::vector<int*> valid_candidate_idx(max_depth);
  for (int j = 1; j < max_depth; ++j)
    valid_candidate_idx[j] = new int[max_candidates_num];
  std::vector<bool> visited_vertices(dg->size(), false);

  std::cout << "Start sequential LFTJ exploration\n";
  uint64_t embedding_cnt = 0;
  auto num = candidates_count[start_vertex];
  for (int i = 0; i < num; i++) {
    auto v0 = candidates[start_vertex][0]; // data vertex at level 0
    embedding[start_vertex] = v0;
    visited_vertices[v0] = true;
    idx_embedding[start_vertex] = i;
    int cur_depth = 1;
    idx[1] = 0;
    generateValidCandidateIndex(1, idx_embedding, bn, bn_count, edge_matrix, order, 
                                temp_buffer, idx_count, valid_candidate_idx[1]);
    while (true) {
      while (idx[cur_depth] < idx_count[cur_depth]) {
        auto valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
        auto u = order[cur_depth];         // query vertex
        auto v = candidates[u][valid_idx]; // data vertex
        if (visited_vertices[v]) {
          idx[cur_depth] += 1;
          continue;
        }
        embedding[u] = v;
        idx_embedding[u] = valid_idx;
        visited_vertices[v] = true;
        idx[cur_depth] += 1;
#ifdef DISTRIBUTION
        begin_count[cur_depth] = embedding_cnt;
        // printf("Cur Depth: %d, v: %u, begin: %zu\n", cur_depth, v, embedding_cnt);
#endif
        if (cur_depth == max_depth - 1) {
#ifdef DISTRIBUTION
          distribution_count_[v] += 1;
#endif
          embedding_cnt += 1;
          visited_vertices[v] = false;
        } else {
          call_count += 1;
          cur_depth += 1;
          idx[cur_depth] = 0;
          generateValidCandidateIndex(cur_depth, idx_embedding, bn, bn_count, edge_matrix, order, 
                                      temp_buffer, idx_count, valid_candidate_idx[cur_depth]);
        }
      }
      cur_depth -= 1;
      auto u = order[cur_depth];
      visited_vertices[embedding[u]] = false;
      if (cur_depth == 0) break;
#ifdef DISTRIBUTION
      distribution_count_[embedding[u]] += embedding_cnt - begin_count[cur_depth];
      // printf("Cur Depth: %d, v: %u, begin: %zu, end: %zu\n", cur_depth, embedding[u], begin_count[cur_depth], embedding_cnt);
#endif
    }
  }
#ifdef DISTRIBUTION
  if (embedding_cnt >= output_limit_num) {
    for (int j = 0; j < max_depth - 1; ++j) {
      auto v = embedding[order[j]];
      distribution_count_[v] += embedding_cnt - begin_count[j];
    }
  }
#endif
  return embedding_cnt;
}

uint64_t QueryPlan::exploreLFTJparallel(const Graph *dg, const Pattern *qg, VertexLists &candidates, 
                                VertexList &candidates_count, VertexList &order,
                                Edges ***edge_matrix, size_t &call_count) {
  int max_depth = qg->size();
  auto start_vertex = order[0];
  auto max_candidates_num = candidates_count[0];
  for (int j = 1; j < max_depth; ++j) {
    auto cur_count = candidates_count[j];
    if (cur_count > max_candidates_num)
      max_candidates_num = cur_count;
  }
  int nt = 1;
  #pragma omp parallel
  {
    nt = omp_get_num_threads();
  }
  VertexLists idxes(nt);
  VertexLists bn_counts(nt);
  VertexLists idx_counts(nt);
  VertexLists embeddings(nt);
  VertexLists idx_embeddings(nt);
  std::vector<int*> temp_buffers(nt);
  std::vector<VertexLists> bns(nt);
  std::vector<std::vector<int*>> valid_candidate_idxes(nt);
  std::vector<std::vector<bool>> masks(nt);
  for (int i = 0; i < nt; i++) {
    masks[i].resize(dg->size());
    idxes[i].resize(max_depth);
    bns[i].resize(max_depth);
    bn_counts[i].resize(max_depth);
    idx_counts[i].resize(max_depth);
    embeddings[i].resize(max_depth);
    idx_embeddings[i].resize(max_depth);
    temp_buffers[i] = new int[max_candidates_num];
    valid_candidate_idxes[i].resize(max_depth);
    std::fill(masks[i].begin(), masks[i].end(), false);
    std::fill(idxes[i].begin(), idxes[i].end(), 0);
    std::fill(bn_counts[i].begin(), bn_counts[i].end(), 0);
    std::fill(idx_counts[i].begin(), idx_counts[i].end(), 0);
    std::fill(embeddings[i].begin(), embeddings[i].end(), 0);
    for (int j = 1; j < max_depth; ++j)
      valid_candidate_idxes[i][j] = new int[max_candidates_num];
    for (int j = 1; j < max_depth; ++j)
      bns[i][j].resize(max_depth);
    generateBN(qg, order, bns[i], bn_counts[i]);
  }

  uint64_t embedding_cnt = 0;
  auto num = candidates_count[start_vertex]; // number of candidates at level 0 (i.e. #v0 for u0)
  std::cout << "Start parallel LFTJ exploration, number of parallel tasks: " << num << "\n";
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:embedding_cnt,call_count)
  for (int i = 0; i < num; i++) {
    auto tid = omp_get_thread_num();
    auto &visited_vertices = masks[tid];
    auto &idx = idxes[tid];
    auto &bn = bns[tid];
    auto &bn_count = bn_counts[tid];
    auto &idx_count = idx_counts[tid];
    auto &embedding = embeddings[tid];
    auto &idx_embedding = idx_embeddings[tid];
    auto &temp_buffer = temp_buffers[tid];
    auto &valid_candidate_idx = valid_candidate_idxes[tid];
    auto v0 = candidates[start_vertex][0]; // data vertex at level 0
    embedding[start_vertex] = v0;
    visited_vertices[v0] = true;
    idx_embedding[start_vertex] = i;
    int cur_depth = 1;
    idx[1] = 0;
    generateValidCandidateIndex(1, idx_embedding, bn, bn_count, edge_matrix, order, 
                                temp_buffer, idx_count, valid_candidate_idx[1]);
#ifdef ENABLE_FAILING_SET
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> ancestors;
    computeAncestor(qg, bn, bn_count, order, ancestors);
    std::vector<std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>> vec_failing_set(max_depth);
    std::unordered_map<vidType, vidType> reverse_embedding;
    reverse_embedding.reserve(MAXIMUM_QUERY_GRAPH_SIZE * 2);
#endif
    while (true) {
      while (idx[cur_depth] < idx_count[cur_depth]) {
        auto valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
        auto u = order[cur_depth];         // query vertex
        auto v = candidates[u][valid_idx]; // data vertex
        if (visited_vertices[v]) {
          idx[cur_depth] += 1;
#ifdef ENABLE_FAILING_SET
          vec_failing_set[cur_depth] = ancestors[u];
          vec_failing_set[cur_depth] |= ancestors[reverse_embedding[v]];
          vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
          continue;
        }
        embedding[u] = v;
        idx_embedding[u] = valid_idx;
        visited_vertices[v] = true;
        idx[cur_depth] += 1;
#ifdef ENABLE_FAILING_SET
        reverse_embedding[v] = u;
#endif
        if (cur_depth == max_depth - 1) {
          embedding_cnt += 1;
          visited_vertices[v] = false;
#ifdef ENABLE_FAILING_SET
          reverse_embedding.erase(embedding[u]);
          vec_failing_set[cur_depth].set();
          vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
#endif
        } else {
          call_count += 1;
          cur_depth += 1;
          idx[cur_depth] = 0;
          generateValidCandidateIndex(cur_depth, idx_embedding, bn, bn_count, edge_matrix, order, 
                                      temp_buffer, idx_count, valid_candidate_idx[cur_depth]);
#ifdef ENABLE_FAILING_SET
          if (idx_count[cur_depth] == 0) {
            vec_failing_set[cur_depth - 1] = ancestors[order[cur_depth]];
          } else {
            vec_failing_set[cur_depth - 1].reset();
          }
#endif
        }
      }
      cur_depth -= 1;
      auto u = order[cur_depth];
      visited_vertices[embedding[u]] = false;
      if (cur_depth == 0) break;
#ifdef ENABLE_FAILING_SET
      reverse_embedding.erase(embedding[u]);
      if (cur_depth != 0) {
        if (!vec_failing_set[cur_depth].test(u)) {
          vec_failing_set[cur_depth - 1] = vec_failing_set[cur_depth];
          idx[cur_depth] = idx_count[cur_depth];
        } else {
          vec_failing_set[cur_depth - 1] |= vec_failing_set[cur_depth];
        }
      }
#endif
    }
  }
#if ENABLE_QFLITER == 1
  delete[] temp_bsr_base1_;
  delete[] temp_bsr_base2_;
  delete[] temp_bsr_state1_;
  delete[] temp_bsr_state2_;
  for (int i = 0; i < max_depth; ++i) {
    for (int j = 0; j < max_depth; ++j) {
      //delete qfliter_bsr_graph_[i][j];
    }
    delete[] qfliter_bsr_graph_[i];
  }
  delete[] qfliter_bsr_graph_;
#endif
  return embedding_cnt;
}

void QueryPlan::generateValidCandidates(const Graph *dg, int depth, VertexList &embedding, 
                                        std::vector<bool> &visited_vertices, 
                                        VertexLists &bn, VertexList &bn_cnt, VertexList &order,
                                        VertexLists &candidates, VertexList &candidates_count,
                                        VertexList &idx_count, VertexLists &valid_candidate) {
  auto u = order[depth];
  idx_count[depth] = 0;
  for (vidType i = 0; i < candidates_count[u]; ++i) {
    auto v = candidates[u][i];
    if (!visited_vertices[v]) {
      bool valid = true;
      for (vidType j = 0; j < bn_cnt[depth]; ++j) {
        auto u_nbr = bn[depth][j];
        auto u_nbr_v = embedding[u_nbr];
        if (!dg->is_connected(v, u_nbr_v)) {
          valid = false;
          break;
        }
      }
      if (valid) {
        valid_candidate[depth][idx_count[depth]++] = v;
      }
    }
  }
}

void QueryPlan::generateValidCandidateIndex(const Graph *dg, int depth, VertexList &embedding, VertexList &idx_embedding,
                                            std::vector<bool> &visited_vertices, VertexLists &bn, VertexList &bn_cnt,
                                            VertexList &order, VertexList &pivot, VertexLists &candidates, Edges ***edge_matrix,
                                            VertexList &idx_count, VertexLists &valid_candidate_index) {
  auto u = order[depth];
  auto pivot_vertex = pivot[depth];
  auto idx_id = idx_embedding[pivot_vertex];
  Edges &edge = *edge_matrix[pivot_vertex][u];
  auto count = edge.offset_[idx_id + 1] - edge.offset_[idx_id];
  vidType *candidate_idx = edge.edge_ + edge.offset_[idx_id];
  vidType valid_candidate_index_count = 0;
  if (bn_cnt[depth] == 0) {
    for (vidType i = 0; i < count; ++i) {
      auto temp_idx = candidate_idx[i];
      auto temp_v = candidates[u][temp_idx];
      if (!visited_vertices[temp_v])
        valid_candidate_index[depth][valid_candidate_index_count++] = temp_idx;
    }
  } else {
    for (vidType i = 0; i < count; ++i) {
      auto temp_idx = candidate_idx[i];
      auto temp_v = candidates[u][temp_idx];
      if (!visited_vertices[temp_v]) {
        bool valid = true;
        for (vidType j = 0; j < bn_cnt[depth]; ++j) {
          auto u_bn = bn[depth][j];
          auto u_bn_v = embedding[u_bn];
          if (!dg->is_connected(temp_v, u_bn_v)) {
            valid = false;
            break;
          }
        }
        if (valid)
          valid_candidate_index[depth][valid_candidate_index_count++] = temp_idx;
      }
    }
  }
  idx_count[depth] = valid_candidate_index_count;
}

void QueryPlan::generateValidCandidateIndex(int depth, VertexList &idx_embedding, VertexLists &bn, VertexList &bn_cnt, 
                                            Edges ***edge_matrix, VertexList &order, int *&temp_buffer,
                                            VertexList &idx_count, int *&valid_candidate_index) {
  auto u = order[depth];
  auto previous_bn = bn[depth][0];
  auto previous_index_id = idx_embedding[previous_bn];
  int valid_candidates_count = 0;
#if ENABLE_QFLITER == 1
  BSRGraph &bsr_graph = *qfliter_bsr_graph_[previous_bn][u];
  BSRSet &bsr_set = bsr_graph.bsrs[previous_index_id];
  if (bsr_set.size_ != 0){
    offline_bsr_trans_uint(bsr_set.base_, bsr_set.states_, bsr_set.size_, valid_candidate_index);
    // represent bsr size
    valid_candidates_count = bsr_set.size_;
  }
  if (bn_cnt[depth] > 0) {
    if (temp_bsr_base1_ == nullptr) { temp_bsr_base1_ = new int[1024 * 1024]; }
    if (temp_bsr_state1_ == nullptr) { temp_bsr_state1_ = new int[1024 * 1024]; }
    if (temp_bsr_base2_ == nullptr) { temp_bsr_base2_ = new int[1024 * 1024]; }
    if (temp_bsr_state2_ == nullptr) { temp_bsr_state2_ = new int[1024 * 1024]; }
    int *res_base_ = temp_bsr_base1_;
    int *res_state_ = temp_bsr_state1_;
    int *input_base_ = temp_bsr_base2_;
    int *input_state_ = temp_bsr_state2_;
    memcpy(input_base_, bsr_set.base_, sizeof(int) * bsr_set.size_);
    memcpy(input_state_, bsr_set.states_, sizeof(int) * bsr_set.size_);
    for (ui i = 1; i < bn_cnt[depth]; ++i) {
      auto current_bn = bn[depth][i];
      ui current_index_id = idx_embedding[current_bn];
      BSRGraph &cur_bsr_graph = *qfliter_bsr_graph_[current_bn][u];
      BSRSet &cur_bsr_set = cur_bsr_graph.bsrs[current_index_id];
      if (valid_candidates_count == 0 || cur_bsr_set.size_ == 0) {
        valid_candidates_count = 0;
        break;
      }
      valid_candidates_count = intersect_qfilter_bsr_b4_v2(cur_bsr_set.base_, cur_bsr_set.states_,
          cur_bsr_set.size_, input_base_, input_state_, valid_candidates_count, res_base_, res_state_);
      std::swap(res_base_, input_base_);
      std::swap(res_state_, input_state_);
    }
    if (valid_candidates_count != 0) {
      valid_candidates_count = offline_bsr_trans_uint(input_base_, input_state_, valid_candidates_count, valid_candidate_index);
    }
  }
#else
  Edges& previous_edge = *edge_matrix[previous_bn][u];
  valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
  int* previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];
  memcpy(valid_candidate_index, previous_candidates, valid_candidates_count * sizeof(int));
  int temp_count;
  for (int i = 1; i < bn_cnt[depth]; ++i) {
    auto current_bn = bn[depth][i];
    Edges& current_edge = *edge_matrix[current_bn][u];
    auto current_index_id = idx_embedding[current_bn];
    auto current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
    int* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];
    SetIntersection::ComputeCandidates(current_candidates, current_candidates_count, 
                                       valid_candidate_index, valid_candidates_count,
                                       temp_buffer, temp_count);
    //std::swap(temp_buffer, valid_candidate_index);
    int *ptr = temp_buffer;
    temp_buffer = valid_candidate_index;
    valid_candidate_index = ptr;
    valid_candidates_count = temp_count;
  }
#endif
  idx_count[depth] = valid_candidates_count;
}

void QueryPlan::generateBN(const Pattern* qg, const VertexList &order, VertexLists &bn, VertexList &bn_count) {
  std::vector<bool> visited_vertices(qg->size(), false);
  visited_vertices[order[0]] = true;
  for (vidType i = 1; i < qg->size(); ++i) {
    auto vertex = order[i];
    auto nbrs_cnt = qg->get_degree(vertex);
    for (vidType j = 0; j < nbrs_cnt; ++j) {
      auto nbr = qg->get_neighbor(vertex, j);
      if (visited_vertices[nbr])
        bn[i][bn_count[i]++] = nbr;
    }
    visited_vertices[vertex] = true;
  }
}

void QueryPlan::checkQueryPlanCorrectness(const Pattern *qg, const VertexList &order, const VertexList &pivot) const {
  int query_vertices_num = qg->size();
  std::vector<bool> visited_vertices(query_vertices_num, false);
  // Check whether each query vertex is in the order.
  for (int i = 0; i < query_vertices_num; ++i) {
    auto vertex = order[i];
    assert(vertex < query_vertices_num && vertex >= 0);
    visited_vertices[vertex] = true;
  }
  for (int i = 0; i < query_vertices_num; ++i) {
    auto vertex = i;
    assert(visited_vertices[vertex]);
  }
  // Check whether the order is connected.
  std::fill(visited_vertices.begin(), visited_vertices.end(), false);
  visited_vertices[order[0]] = true;
  for (int i = 1; i < query_vertices_num; ++i) {
    auto vertex = order[i];
    auto pivot_vertex = pivot[i];
    assert(qg->is_connected(vertex, pivot_vertex));
    assert(visited_vertices[pivot_vertex]);
    visited_vertices[vertex] = true;
  }
}

void QueryPlan::printQueryPlan(const Pattern *qg, const VertexList &order) const {
  printf("Query Plan: ");
  for (vidType i = 0; i < qg->size(); ++i)
    printf("%u, ", order[i]);
  printf("\n");
  printf("%u: N/A\n", order[0]);
  for (vidType i = 1; i < qg->size(); ++i) {
    auto end_vertex = order[i];
    printf("%u: ", end_vertex);
    for (vidType j = 0; j < i; ++j) {
      auto begin_vertex = order[j];
      if (qg->is_connected(begin_vertex, end_vertex)) {
        printf("R(%u, %u), ", begin_vertex, end_vertex);
      }
    }
    printf("\n");
  }
}

void QueryPlan::printSimplifiedQueryPlan(const Pattern *qg, const VertexList &order) const {
  printf("Query Plan: ");
  for (vidType i = 0; i < qg->size(); ++i)
    printf("%u ", order[i]);
  printf("\n");
}

