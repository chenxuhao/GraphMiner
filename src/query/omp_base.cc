#include "graph.h"
#include "query_plan.h"

void extend_vertex(int level, int k, Graph &g, Pattern &p,
                   VertexList &stack, 
                   std::vector<VertexSet> &frontiers,
                   std::vector<VertexSet> &buffers,
                   uint64_t& counter);
 
void QuerySolver(Graph &g, Pattern &p, uint64_t &total, bool, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Graph Querying (" << num_threads << " threads)\n";
  p.analyze();
  uint64_t counter = 0;
  int k = p.size();

  Timer t;
  t.Start();
  #pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    VertexList stack(k);
    std::vector<VertexSet> frontiers(k-2);
    std::vector<VertexSet> buffers(k-2);
    auto l0 = g.get_vlabel(v0);
    if (l0 != p.get_vlabel(0)) continue;
    stack[0] = v0;
    auto y0 = g.N(v0);
    for (auto v1 : y0) {
      auto l1 = g.get_vlabel(v1);
      if (l1 != p.get_vlabel(1)) continue; 
      stack[1] = v1;
      extend_vertex(2, k, g, p, stack, frontiers, buffers, counter);
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

void extend_vertex(int level, int k, Graph &g, Pattern &p,
                   VertexList &stack,
                   std::vector<VertexSet> &frontiers, 
                   std::vector<VertexSet> &buffers, 
                   uint64_t& counter) {
  auto n = p.get_num_operators(level);
  assert(n>=1);
  auto v = stack[p.get_operand(level, 0)];
  auto u = stack[p.get_operand(level, 1)];
  for (int i = 0; i < n-1; i++)
    buffers[i].clear();

  if (level == k-1) { // last level
    uint64_t local_count = 0;
    if (n == 1) { // last operator; count
      if (p.get_setop(level, 0) == SET_INTERSECTION) {
        local_count += g.intersect_num(v, u, p.get_vlabel(level));
      } else {
        local_count += g.difference_num_edgeinduced(v, u, p.get_vlabel(level));
      }
    } else {
      if (p.get_setop(level, 0) == SET_INTERSECTION) {
        g.intersect_set(v, u, p.get_vlabel(level), buffers[0]);
      } else {
        g.difference_set_edgeinduced(v, u, p.get_vlabel(level), buffers[0]);
      }
    }
    for (int i = 1; i < n; i++) {
      u = stack[p.get_operand(level, i+1)];
      if (i == n-1) { // last operator; count
        if (p.get_setop(level, i) == SET_INTERSECTION)
          local_count = g.intersect_num(buffers[i-1], u, p.get_vlabel(level));
        else
          local_count = g.difference_num_edgeinduced(buffers[i-1], u, p.get_vlabel(level));
      } else { // list
        if (p.get_setop(level, i) == SET_INTERSECTION) {
          g.intersect_set(buffers[i-1], u, p.get_vlabel(level), buffers[i]);
        } else {
          g.difference_set_edgeinduced(buffers[i-1], u, p.get_vlabel(level), buffers[i]);
        }
      }
    }
    counter += local_count;
    return;
  }

  frontiers[level-2].clear();
  if (p.get_setop(level, 0) == SET_INTERSECTION) {
    if (n == 1)
      g.intersect_set(v, u, p.get_vlabel(level), frontiers[0]);
    else
      g.intersect_set(v, u, p.get_vlabel(level), buffers[0]);
  } else {
    if (n == 1)
      g.difference_set_edgeinduced(v, u, p.get_vlabel(level), frontiers[0]);
    else
      g.difference_set_edgeinduced(v, u, p.get_vlabel(level), buffers[0]);
  }
  int i = 1;
  for (; i < n-1; i++) {
    u = stack[p.get_operand(level, i+1)];
    if (p.get_setop(level, i) == SET_INTERSECTION) {
      g.intersect_set(buffers[i-1], u, p.get_vlabel(level), buffers[i]);
    } else {
      g.difference_set_edgeinduced(buffers[i-1], u, p.get_vlabel(level), buffers[i]);
    }
  }
  if (n > 1) {
    if (p.get_setop(level, i) == SET_INTERSECTION) {
      g.intersect_set(buffers[i-1], u, p.get_vlabel(level), frontiers[level-2]);
    } else {
      g.difference_set_edgeinduced(buffers[i-1], u, p.get_vlabel(level), frontiers[level-2]);
    }
  }
  for (i = 0; i < frontiers[level-2].size(); i++) {
    auto v = frontiers[level-2][i];
    stack[level] = v;
    extend_vertex(level+1, k, g, p, stack, frontiers, buffers, counter);
  }
}

