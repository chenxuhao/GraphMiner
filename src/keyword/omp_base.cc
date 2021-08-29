#include "gks.h"
#include "subgraph.h"

inline bool filter(Graph &g, Subgraph sg, Keywords kw, int max_size) {
  if (sg.size() > size_t(max_size)) return false;
  for (auto label : kw)
    if (sg.has_more_than_one(g, label)) return false;
  return true;
}

inline bool is_match(Graph &g, Subgraph sg, Keywords kws) {
  for (auto label : kws)
    if (!sg.has_only_one(g, label)) return false;
  int n = sg.size();
  for (int i = 1; i < n; i++) {
    auto v = sg.get_vertex(i);
    auto a = g.get_vlabel(v);
    if (kws.contains(a)) continue;
    if (sg.is_connected_without(i))
      return false;
  }
  return true;
}

void extend_vertex(int max_size, Keywords kws, Graph &g, Subgraph &sg, uint64_t &count);

void GksSolver(Graph &g, int k, int nlables, Keywords keywords, uint64_t &total, int, int) {
  assert(k >= 2);
  assert(keywords.size()<=size_t(k));
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Graph Keyword Search (" << num_threads << " threads)\n";
  uint64_t counter = 0;
  Timer t;
  t.Start();
  #pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    if (!keywords.contains(g.get_vlabel(v0))) continue;
    Subgraph sg(v0);
    for (vidType v1 : g.N(v0)) {
      if (g.get_vlabel(v0) == g.get_vlabel(v1)) continue;
      sg.push_back(v1);
      if (k == 2 && keywords.contains(g.get_vlabel(v1)))
        counter ++;
      else extend_vertex(k, keywords, g, sg, counter);
      sg.pop_back();
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

void extend_vertex(int max_size, Keywords kws, Graph &g, Subgraph &sg, uint64_t &count) {
  int n = sg.size();
  if (n == max_size) {
    if (is_match(g, sg, kws))
      count ++;
    return;
  }
  for (int i = 0; i < n; i++) {
    auto v = sg.get_vertex(i);
    for (auto u : g.N(v)) {
      if (!sg.is_canonical(g, u, i)) continue;
      if (filter(g, sg, kws, max_size)) {
        sg.push(u, i, g);
        extend_vertex(max_size, kws, g, sg, count);
        sg.pop();
      }
    }
  }
}

