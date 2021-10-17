// This code is modified from DistGraph:
// https://github.com/zakimjz/DistGraph.git
#pragma once
#include "dfscode.h"
#include "domain_support.h"
typedef DFSCode Pattern;
typedef DFS PatternElement;
typedef std::map<unsigned, std::map<unsigned, unsigned>> Map2D;

unsigned support(Graph &g, BaseEdgeEmbeddingList& emb_list, Pattern& pattern) {
  Map2D vid_counts;
  for (auto cur = emb_list.begin(); cur != emb_list.end(); ++cur) {
    BaseEdgeEmbedding* emb_ptr = &(*cur);
    size_t index = pattern.size() - 1;
    while (emb_ptr != NULL) {
      auto from = pattern[index].from;
      auto to   = pattern[index].to;
      auto src  = g.get_src(emb_ptr->edge);
      auto dst  = g.get_dst(emb_ptr->edge);
      if (to > from)
        vid_counts[to][dst]++; // forward edge
      if (!emb_ptr->prev)
        vid_counts[from][src]++; // last element
      emb_ptr = emb_ptr->prev;
      index--;
    }
  }
  unsigned min = 0xffffffff;
  for (auto it = vid_counts.begin(); it != vid_counts.end(); it++)
    if ((it->second).size() < min) min = (it->second).size();
  if (min == 0xffffffff) min = 0;
  return min;
}

inline bool is_frequent(Graph &g, BaseEdgeEmbeddingList& emb_list, Pattern& pattern, unsigned threshold) {
  if (emb_list.size() < threshold) return false;
  DomainSupport ds(pattern.size() + 1);
  ds.set_threshold(threshold);
  unsigned emb_id = 0;
  for (auto cur = emb_list.begin(); cur != emb_list.end(); ++cur) {
    BaseEdgeEmbedding* emb_ptr = &(*cur);
    size_t index               = pattern.size() - 1;
    while (emb_ptr != NULL) {
      auto from = pattern[index].from;
      auto to   = pattern[index].to;
      auto src  = g.get_src(emb_ptr->edge);
      auto dst  = g.get_dst(emb_ptr->edge);
      if (!ds.has_domain_reached_support(to) && to > from)
        ds.add_vertex(to, dst); // forward edge
      if (!ds.has_domain_reached_support(from) && !emb_ptr->prev)
        ds.add_vertex(from, src); // last element
      emb_ptr = emb_ptr->prev;
      index--;
    }
    emb_id++;
    if (emb_id >= threshold)
      ds.set_frequent();
    if (ds.is_frequent())
      return true;
  }
  return false;
}

inline bool subgraph_is_min(Graph &g, Pattern& orig_pattern, Pattern& pattern, CGraph& cgraph, BaseEdgeEmbeddingList& emb_list) {
  const RMPath& rmpath = pattern.buildRMPath();
  auto minlabel        = pattern[0].fromlabel;
  auto maxtoc          = pattern[rmpath[0]].to;
  // backward
  bool found   = false;
  vidType newto = 0;
  BaseEdgeEmbeddingList emb_list_bck;
  for (size_t i = rmpath.size() - 1; i >= 1; --i) {
    for (size_t j = 0; j < emb_list.size(); ++j) {
      BaseEdgeEmbedding* cur = &emb_list[j];
      History history(g, cur);
      auto e1 = history[rmpath[i]];
      auto e2 = history[rmpath[0]];
      if (e1 == e2) continue;
      auto e1_src = g.get_src(e1);
      auto e1_dst = g.get_dst(e1);
      auto e2_dst = g.getEdgeDst(e2);
      for (auto e = cgraph[e2_dst].edges.begin(); e != cgraph[e2_dst].edges.end(); ++e) {
        auto e_dst = e->dst;
        if (history.hasEdge(g, e->src, e_dst)) continue;
        if ((e_dst == e1_src) && (cgraph[e1_dst].label <= cgraph[e2_dst].label)) {
          emb_list_bck.push(2, &(*e), cur);
          newto = pattern[rmpath[i]].from;
          found = true;
          break;
        }
      }
    }
  }
  if (found) {
    pattern.push(maxtoc, newto, label_t(-1), 0, label_t(-1));
    auto size = pattern.size() - 1;
    if (orig_pattern[size] != pattern[size]) return false;
    return subgraph_is_min(g, orig_pattern, pattern, cgraph, emb_list_bck);
  }

  // forward
  bool flg = false;
  vidType newfrom = 0;
  EmbeddingLists1D emb_lists_fwd;
  for (size_t n = 0; n < emb_list.size(); ++n) {
    BaseEdgeEmbedding* cur = &emb_list[n];
    History history(g, cur);
    auto e2 = history[rmpath[0]];
    auto e2_dst = g.getEdgeDst(e2);
    for (auto e = cgraph[e2_dst].edges.begin(); e != cgraph[e2_dst].edges.end(); ++e) {
      auto e_dst = e->dst;
      if (minlabel > cgraph[e_dst].label || history.hasVertex(e_dst))
        continue;
      if (flg == false) {
        flg     = true;
        newfrom = maxtoc;
      }
      emb_lists_fwd[cgraph[e_dst].label].push(2, &(*e), cur);
    }
  }
  for (size_t i = 0; !flg && i < rmpath.size(); ++i) {
    for (size_t n = 0; n < emb_list.size(); ++n) {
      BaseEdgeEmbedding* cur = &emb_list[n];
      History history(g, cur);
      auto e1 = history[rmpath[i]];
      auto e1_src = g.get_src(e1);
      auto e1_dst = g.get_dst(e1);
      for (auto e = cgraph[e1_src].edges.begin(); e != cgraph[e1_src].edges.end(); ++e) {
        auto dst = e->dst;
        auto& v  = cgraph[dst];
        if (e1_dst == dst || minlabel > v.label || history.hasVertex(dst))
          continue;
        if (cgraph[e1_dst].label <= v.label) {
          if (flg == false) {
            flg     = true;
            newfrom = pattern[rmpath[i]].from;
          }
          emb_lists_fwd[v.label].push(2, &(*e), cur);
        }
      }
    }
  }
  if (flg) {
    auto tolabel = emb_lists_fwd.begin();
    pattern.push(newfrom, maxtoc + 1, (label_t)-1, 0, tolabel->first);
    auto size = pattern.size() - 1;
    if (orig_pattern[size] != pattern[size])
      return false;
    return subgraph_is_min(g, orig_pattern, pattern, cgraph, tolabel->second);
  }
  return true;
}

inline bool is_canonical(Graph &g, Pattern& pattern) {
  if (pattern.size() == 1) return true;
  CGraph graph_is_min; // canonical graph
  pattern.toGraph(graph_is_min);
  Pattern dfscode_is_min;
  EmbeddingLists2D emb_lists;
  for (size_t vid = 0; vid < graph_is_min.size(); ++vid) {
    auto vlabel = graph_is_min[vid].label;
    for (auto e = graph_is_min[vid].edges.begin(); e != graph_is_min[vid].edges.end(); ++e) {
      auto ulabel = graph_is_min[e->dst].label;
      if (vlabel <= ulabel)
        emb_lists[vlabel][ulabel].push(2, &(*e), 0);
    }
  }
  auto fromlabel = emb_lists.begin();
  auto tolabel   = fromlabel->second.begin();
  dfscode_is_min.push(0, 1, fromlabel->first, 0, tolabel->first);
  return subgraph_is_min(g, pattern, dfscode_is_min, graph_is_min, tolabel->second);
}

inline bool toAdd(Graph &g, BaseEdgeEmbeddingList& emb_list, Pattern& pattern, unsigned threshold) {
  if (pattern.size() == 1) return true;
  if (!is_frequent(g, emb_list, pattern, threshold)) return false;
  if (pattern.size() == 2) {
    if (pattern[1].from == 1) {
      if (pattern[0].fromlabel <= pattern[1].tolabel) return true;
    } else {
      assert(pattern[1].from == 0);
      if (pattern[0].fromlabel == pattern[0].tolabel) return false;
      if (pattern[0].tolabel == pattern[1].tolabel && pattern[0].fromlabel < pattern[1].tolabel) return true;
      if (pattern[0].tolabel <  pattern[1].tolabel) return true;
    }
    return false;
  }
  return is_canonical(g, pattern);
}

