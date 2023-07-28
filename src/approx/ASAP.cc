#include "graph.h"
#include "ASAP.h"

bool get_closure(Graph &g, vidType v1, vidType v2, int depth) {
    if(depth == 0) {
        return v1 == v2;
    } else {
        for(auto v : g.N(v1)) {
            if(get_closure(g, v1,v ,depth-1)) {
                return true;
            }
        }
    }
    return false;
}

output sample_out_neighbor_edge(Graph &g, vidType v0, vidType v1) {
    if(g.N(v1).size() > 0) {
      // get a neighbor of l1, vertex l2
      // 1/c prob of choosing some neighbor
      auto l2 = v0;
      while(l2 == v0) {
        l2 = g.N(v1)[rand() % g.N(v1).size()];
      }
      return {l2, g.N(v1).size()};
    }

    return {-1,-1};
}

output sample_edge(Graph &g) {
    g.init_simple_edgelist(); // will skip if already init
    eidType randE = rand() % g.E();

    return {g.get_src(randE), g.get_dst(randE), g.E()};
}