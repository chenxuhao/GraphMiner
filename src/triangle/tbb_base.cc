// Copyright 2023 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <tbb/parallel_for.h>
#include <oneapi/tbb/info.h>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
//#include <tbb/task_scheduler_init.h>

void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = tbb::info::default_concurrency();
  //int num_threads = tbb::task_scheduler_init::default_num_threads();
  //int num_threads = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
  std::cout << "TBB TC (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  std::vector<uint64_t> counters(num_threads, 0);
  tbb::parallel_for(tbb::blocked_range<vidType>(0, g.V(), 1),
    [&g, &counters](tbb::blocked_range<vidType> &r) {
    auto tid = tbb::this_task_arena::current_thread_index();
    auto &counter = counters.at(tid);
    for (vidType u = r.begin(); u < r.end(); u ++) {
      auto adj_u = g.N(u);
      for (auto v : adj_u) {
        counter += (uint64_t)intersection_num(adj_u, g.N(v));
      }
    }
  });
  for (int i = 0; i < num_threads; i++)
    total += counters[i];
  t.Stop();
  std::cout << "runtime [tbb_base] = " << t.Seconds() << " sec\n";
  return;
}

