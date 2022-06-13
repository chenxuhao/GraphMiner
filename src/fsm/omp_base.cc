// Copyright 2020
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "canonical.h"

typedef std::pair<label_t, label_t> InitPattern;
typedef std::map<InitPattern, DomainSupport*> InitMap;
typedef std::vector<PatternElement> InitPatternQueue;

inline InitPattern get_init_pattern(label_t src_label, label_t  dst_label) {
  if (src_label <= dst_label)
    return std::make_pair(src_label, dst_label);
  else
    return std::make_pair(dst_label, src_label);
}

void dfs_extend(int level, int max_size, int minsup, Graph &g, BaseEdgeEmbeddingList& emb_list, Pattern& pattern, int &total);

void FsmSolver(Graph &g, int k, int minsup, int &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP FSM (" << num_threads << " threads)\n";
  InitPatternQueue init_queue; // frequent single-edge patterns
  std::vector<InitMap> init_pattern_maps(num_threads);
  g.init_edgelist();
  std::vector<Timer> timers(6);

  Timer t;
  t.Start();
  timers[0].Start();
  #pragma omp parallel for
  for (eidType eid = 0; eid < g.E(); eid++) {
    auto v = g.get_src(eid);
    auto u = g.get_dst(eid);
    if (!g.is_freq_vertex(v, minsup) || !g.is_freq_vertex(u, minsup)) continue;
    auto v_label = g.get_vlabel(v);
    auto u_label = g.get_vlabel(u);
    auto tid = omp_get_thread_num();
    auto& lmap = init_pattern_maps[tid];
    if (v_label <= u_label) {
      auto key = get_init_pattern(v_label, u_label);
      if (lmap.find(key) == lmap.end()) {
        lmap[key] = new DomainSupport(2);
        lmap[key]->set_threshold(minsup);
      }
      lmap[key]->add_vertex(0, v);
      lmap[key]->add_vertex(1, u);
    }
  }
  timers[0].Stop();

  timers[1].Start();
  // merge thread-private maps into a single map
  auto& init_map = init_pattern_maps[0];
  for (auto i = 1; i < num_threads; i++) {
    for (auto element : init_pattern_maps[i]) {
      DomainSupport* support = element.second;
      if (init_map.find(element.first) == init_map.end()) {
        init_map[element.first] = support;
      } else {
        for (unsigned i = 0; i < 2; i++) {
          if (!init_map[element.first]->has_domain_reached_support(i)) {
            if (support->has_domain_reached_support(i))
              init_map[element.first]->set_domain_frequent(i);
            else
              init_map[element.first]->add_vertices(i, support->domain_sets[i]);
          }
        }
      }
    }
  }
  //int single_edge_patterns = init_map.size();
  timers[1].Stop();

  timers[2].Start();
  // classify all the single-edge embeddings
  int num_embeddings = 0;
  int num_freq_embeddings = 0;
  EmbeddingLists2D init_emb_lists;
  for (eidType eid = 0; eid < g.E(); eid++) {
    auto v = g.get_src(eid);
    auto u = g.get_dst(eid);
    if (!g.is_freq_vertex(v, minsup) || !g.is_freq_vertex(u, minsup)) continue;
    auto src_label = g.getData(v);
    auto dst_label = g.getData(u);
    if (src_label <= dst_label) {
      num_embeddings ++;
      auto key = get_init_pattern(src_label, dst_label);
      if (init_map[key]->get_support()) {
        num_freq_embeddings ++;
        init_emb_lists[src_label][dst_label].push(2, eid, 0);
      }
    }
  }
  timers[2].Stop();

  timers[3].Start();
  int num_freq_patterns = 0;
  //#pragma omp parallel for reduction(+:num_freq_patterns)
  for (auto p : init_map) {
    if (p.second->get_support())
      num_freq_patterns += 1;
  }
  init_queue.resize(num_freq_patterns);
  timers[3].Stop();

  timers[4].Start();
  num_freq_patterns = 0;
  //#pragma omp parallel for
  for (auto p : init_map) {
    // if the pattern is frequent, add it to the pattern queue
    if (p.second->get_support()) {
      auto src_label = p.first.first;
      auto dst_label = p.first.second;
      PatternElement dfs(0, 1, src_label, 0, dst_label);
      int id = __sync_fetch_and_add(&num_freq_patterns, 1);
      init_queue[id] = dfs;
    }
  }
  timers[4].Stop();
  std::cout << "Number of frequent single-edge patterns: " << num_freq_patterns << "\n";
  std::cout << "Number of frequent single-edge embeddings: " << num_freq_embeddings << "\n";

  timers[5].Start();
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:total)
  for (size_t i = 0; i < init_queue.size(); i++) {
    auto& p = init_queue[i];
    Pattern pattern;
    auto src_lab = p.fromlabel;
    auto dst_lab = p.tolabel;
    pattern.push(0, 1, src_lab, 0, dst_lab); // current pattern
    dfs_extend(1, k, minsup, g, init_emb_lists[src_lab][dst_lab], pattern, total);
    pattern.pop();
  } 
  timers[5].Stop();
  t.Stop();
  std::cout << "runtime [0] = " << timers[0].Seconds() << " sec\n";
  std::cout << "runtime [1] = " << timers[1].Seconds() << " sec\n";
  std::cout << "runtime [2] = " << timers[2].Seconds() << " sec\n";
  std::cout << "runtime [3] = " << timers[3].Seconds() << " sec\n";
  std::cout << "runtime [4] = " << timers[4].Seconds() << " sec\n";
  std::cout << "runtime [5] = " << timers[5].Seconds() << " sec\n";
  std::cout << "runtime [omp_base] = " << t.Seconds() <<  " sec\n";
  return;
}

void dfs_extend(int level, int max_size, int minsup, Graph &g, BaseEdgeEmbeddingList& emb_list, Pattern& pattern, int &total_num) {
  total_num ++; // list frequent patterns here!!!
  if (level == max_size) return;
  const RMPath& rmpath = pattern.buildRMPath(); // build the right-most path of this pattern
  auto minlabel = pattern[0].fromlabel;
  auto maxtoc   = pattern[rmpath[0]].to; // right-most vertex
  EmbeddingLists2D emb_lists_fwd;
  EmbeddingLists1D emb_lists_bck;
  for (size_t emb_id = 0; emb_id < emb_list.size(); ++emb_id) {
    BaseEdgeEmbedding* cur = &emb_list[emb_id];
    unsigned emb_size      = cur->num_vertices;
    History history(g, cur);
    auto e2 = history[rmpath[0]];
    auto e2_dst = g.getEdgeDst(e2);
    // backward extension
    for (size_t i = rmpath.size() - 1; i >= 1; --i) {
      auto e1 = history[rmpath[i]];
      if (e1 == e2) continue;
      auto src = e2_dst;
      auto src_label = g.getData(src);
      auto u = g.get_src(e1);
      auto v = g.get_dst(e1);
      auto vlabel = g.getData(v);
      for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
        auto dst = g.getEdgeDst(e);
        if (history.hasEdge(g, src, dst)) continue;
        if (dst == u && vlabel <= src_label) {
          auto w = pattern[rmpath[i]].from;
          emb_lists_bck[w].push(emb_size, e, cur);
          break;
        }
      }
    }
    // pure forward extension
    for (auto e = g.edge_begin(e2_dst); e != g.edge_end(e2_dst); e++) {
      auto dst = g.getEdgeDst(e);
      auto dst_label = g.getData(dst);
      if (minlabel > dst_label || history.hasVertex(dst)) continue;
      //auto vlabel = g.getData(edge_list.get_edge(e).dst);
      emb_lists_fwd[maxtoc][dst_label].push(emb_size+1, e, cur);
    }
    // backtracked forward extension
    for (size_t i = 0; i < rmpath.size(); ++i) {
      auto e1 = history[rmpath[i]];
      auto e1_dst = g.getEdgeDst(e1);
      auto src_label = g.getData(e1_dst);
      auto u = g.get_src(e1);
      for (auto e = g.edge_begin(u); e != g.edge_end(u); e++) {
        auto dst = g.getEdgeDst(e);
        auto dst_label = g.getData(dst);
        if (g.getEdgeDst(e1) == dst || minlabel > dst_label || history.hasVertex(dst))
          continue;
        if (src_label <= dst_label) {
          auto w = pattern[rmpath[i]].from;
          //auto vlabel = g.getData(edge_list.get_edge(e).dst);
          emb_lists_fwd[w][dst_label].push(emb_size+1, e, cur);
        }
      }
    }
  }
  std::vector<PatternElement> pattern_list;
  for (auto to = emb_lists_bck.begin(); to != emb_lists_bck.end(); ++to) {
    PatternElement dfs(maxtoc, to->first, (label_t)-1, 0, (label_t)-1);
    Pattern new_pattern = pattern;
    new_pattern.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel);
    //if (support(g, emb_lists_bck[dfs.to], new_pattern) >= minsup && is_min(new_pattern)) {
    if (toAdd(g, emb_lists_bck[dfs.to], new_pattern, minsup)) {
      pattern_list.push_back(dfs);
    }
  }
  for (auto from = emb_lists_fwd.rbegin(); from != emb_lists_fwd.rend(); ++from) {
    for (auto tolabel = from->second.begin(); tolabel != from->second.end(); ++tolabel) {
      PatternElement dfs(from->first, maxtoc + 1, (label_t)-1, 0, tolabel->first);
      Pattern new_pattern = pattern;
      new_pattern.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel);
      //if (support(g, emb_lists_fwd[dfs.from][dfs.tolabel], new_pattern) >= minsup && is_min(new_pattern)) {
      if (toAdd(g, emb_lists_fwd[dfs.from][dfs.tolabel], new_pattern, minsup)) {
        pattern_list.push_back(dfs);
      }
    }
  }
  for (auto dfs : pattern_list) {
    pattern.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel); // update the pattern
    if (dfs.is_backward())
      dfs_extend(level + 1, max_size, minsup, g, emb_lists_bck[dfs.to], pattern, total_num);
    else
      dfs_extend(level + 1, max_size, minsup, g, emb_lists_fwd[dfs.from][dfs.tolabel], pattern, total_num);
    pattern.pop();
  }
}

