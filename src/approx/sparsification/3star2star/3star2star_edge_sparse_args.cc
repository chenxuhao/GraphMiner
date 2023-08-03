#include "graph.h"
#include "pattern.hh"

void EdgeSample(Graph &g, float p) {
  std::cout << "|e| before sampling " << g.E() << "\n";
  g.edge_sparsify(p);
  std::cout <<  "|e| after sampling " << g.E() << "\n";
}

void automine_3star2star(Graph &g, std::vector<std::vector<uint64_t>> &global_counters) {
  #pragma omp parallel
  {
    auto &counter = global_counters.at(omp_get_thread_num());
    #pragma omp for schedule(dynamic,1)
    for(vidType v0 = 0; v0 < g.V(); v0++) {
      VertexSet y0 = g.N(v0);
      VertexSet y0f0 = bounded(y0,v0);
      for(vidType idx1 = 0; idx1 < y0.size(); idx1++) {
        vidType v1 = y0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        VertexSet y0n1f1 = difference_set(y0, y1, v1);
        for(vidType idx2 = 0; idx2 < y0n1f1.size(); idx2++) {
          vidType v2 = y0n1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          VertexSet counter0 = difference_set(y0n1f1, y2, v2); // the size of this set is the num 3 stars?

          for(vidType idx3 = 0; idx3 < counter0.size(); idx3++) {
             vidType v3 = y0n1f1.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[0] = difference_num(counter0, y3, v3);
          }
        }
      }
    }
}
}

void MotifSolver(Graph &g, int k, std::vector<uint64_t> &total, int, int, vector<float> args) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Motif solver (" << num_threads << " threads) ...\n";

  EdgeSample(g, args[0]);
  int num_patterns = num_possible_patterns[k];
  std::vector<std::vector<uint64_t>> global_counters(num_threads);
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &local_counters = global_counters[tid];
    local_counters.resize(num_patterns);
    std::fill(local_counters.begin(), local_counters.end(), 0);
  } 
  Timer t;
  t.Start();
  automine_3star2star(g, global_counters);
  for (int tid = 0; tid < num_threads; tid++)
    for (int pid = 0; pid < num_patterns; pid++)
      total[pid] += global_counters[tid][pid]; //todo add multiplication factor

  //star (3), chain(3), 3-loop-out(4), box(4), semi-clique(5), clique(6) 
  printf("p: %f\n", args[0]);
  for(int pid = 0; pid < num_patterns; pid++) { 
    total[pid] = total[pid]*(args[0]*args[0]*args[0]); //all nodes have same color
  }
  
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << "\n";
}
