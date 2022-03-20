
void automine_3motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters) {
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
        counter[0] += difference_num(y0, y1, v1);
      }
      for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
        vidType v1 = y0f0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        counter[1] += intersection_num(y0f0, y1, v1);
      }
    }
  }
}

void automine_4motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters) {
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
          counter[0] += difference_num(y0n1f1, y2, v2);
        }
      }
      for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
        vidType v1 = y0f0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        VertexSet y0y1 = intersection_set(y0, y1);
        VertexSet y0f0y1f1 = intersection_set(y0f0, y1, v1);
        VertexSet n0y1; difference_set(n0y1,y1, y0);
        VertexSet n0f0y1; difference_set(n0f0y1,y1, y0);
        VertexSet y0n1 = difference_set(y0, y1);
        VertexSet y0f0n1f1 = difference_set(y0f0, y1, v1);
        for(vidType idx2 = 0; idx2 < y0y1.size(); idx2++) {
          vidType v2 = y0y1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter[4] += difference_num(y0y1, y2, v2);
          VertexSet n0n1y2; counter[2] += difference_num(difference_set(n0n1y2,y2, y0), y1);
        }
        for(vidType idx2 = 0; idx2 < y0f0y1f1.size(); idx2++) {
          vidType v2 = y0f0y1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter[5] += intersection_num(y0f0y1f1, y2, v2);
        }
        for(vidType idx2 = 0; idx2 < y0n1.size(); idx2++) {
          vidType v2 = y0n1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter[1] += difference_num(n0y1, y2);
        }
        for(vidType idx2 = 0; idx2 < y0f0n1f1.size(); idx2++) {
          vidType v2 = y0f0n1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter[3] += intersection_num(n0f0y1, y2, v0);
        }
      }
    }
  }
}

void automine_5motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters) {
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
        VertexSet y0y1 = intersection_set(y0, y1);
        VertexSet n0y1; difference_set(n0y1,y1, y0);
        VertexSet n0y1f0 = bounded(n0y1,v0);
        VertexSet y0n1 = difference_set(y0, y1);
        VertexSet y0n1f1 = bounded(y0n1,v1);
        for(vidType idx2 = 0; idx2 < n0y1f0.size(); idx2++) {
          vidType v2 = n0y1f0.begin()[idx2];
          VertexSet y2 = g.N(v2);
          VertexSet y0n1y2 = intersection_set(y0n1, y2);
          VertexSet y0n1y2f1 = bounded(y0n1y2,v1);
          VertexSet y0n1f1y2 = intersection_set(y0n1f1, y2);
          for(vidType idx3 = 0; idx3 < y0n1y2.size(); idx3++) {
            vidType v3 = y0n1y2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            VertexSet n0n1n2y3; counter[3] += difference_num(difference_set(n0n1n2y3,difference_set(n0n1n2y3,y3, y0), y1), y2);
          }
          for(vidType idx3 = 0; idx3 < y0n1y2f1.size(); idx3++) {
            vidType v3 = y0n1y2f1.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[5] += difference_num(y0n1f1y2, y3, v3);
          }
        }
        for(vidType idx2 = 0; idx2 < y0n1.size(); idx2++) {
          vidType v2 = y0n1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          VertexSet n0y1n2 = difference_set(n0y1, y2);
          VertexSet y0n1n2f2 = difference_set(y0n1, y2, v2);
          for(vidType idx3 = 0; idx3 < y0n1n2f2.size(); idx3++) {
            vidType v3 = y0n1n2f2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[1] += difference_num(n0y1n2, y3);
          }
        }
        for(vidType idx2 = 0; idx2 < y0n1f1.size(); idx2++) {
          vidType v2 = y0n1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          VertexSet y0y1y2 = intersection_set(y0y1, y2);
          VertexSet n0n1y2; difference_set(n0n1y2,difference_set(n0n1y2,y2, y0), y1);
          VertexSet n0y1n2 = difference_set(n0y1, y2);
          VertexSet y0n1f1n2f2 = difference_set(y0n1f1, y2, v2);
          for(vidType idx3 = 0; idx3 < y0y1y2.size(); idx3++) {
            vidType v3 = y0y1y2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            VertexSet n0n1n2y3; counter[4] += difference_num(difference_set(n0n1n2y3,difference_set(n0n1n2y3,y3, y0), y1), y2);
          }
          for(vidType idx3 = 0; idx3 < n0y1n2.size(); idx3++) {
            vidType v3 = n0y1n2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[8] += difference_num(n0n1y2, y3);
          }
          for(vidType idx3 = 0; idx3 < y0n1f1n2f2.size(); idx3++) {
            vidType v3 = y0n1f1n2f2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[0] += difference_num(y0n1f1n2f2, y3, v3);
          }
        }
      }
      for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
        vidType v1 = y0f0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        VertexSet y0y1 = intersection_set(y0, y1);
        VertexSet y0f0y1f1 = intersection_set(y0f0, y1, v1);
        VertexSet n0y1; difference_set(n0y1,y1, y0);
        VertexSet n0f0y1; difference_set(n0f0y1,y1, y0);
        VertexSet y0n1 = difference_set(y0, y1);
        VertexSet y0f0n1f1 = difference_set(y0f0, y1, v1);
        for(vidType idx2 = 0; idx2 < y0y1.size(); idx2++) {
          vidType v2 = y0y1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          VertexSet n0y1y2 = intersection_set(n0y1, y2);
          VertexSet n0f0y1y2 = intersection_set(n0f0y1, y2);
          VertexSet y0n1y2 = intersection_set(y0n1, y2);
          VertexSet y0f0n1y2f1 = intersection_set(y0f0n1f1, y2, v1);
          VertexSet y0y1n2 = difference_set(y0y1, y2);
          VertexSet y0y1n2f2 = bounded(y0y1n2,v2);
          VertexSet n0n1y2; difference_set(n0n1y2,difference_set(n0n1y2,y2, y0), y1);
          VertexSet n0n1y2f0 = bounded(n0n1y2,v0);
          VertexSet n0f0n1y2; difference_set(n0f0n1y2,difference_set(n0f0n1y2,y2, y0), y1);
          VertexSet n0y1n2 = difference_set(n0y1, y2);
          VertexSet y0n1n2 = difference_set(y0n1, y2);
          for(vidType idx3 = 0; idx3 < y0n1y2.size(); idx3++) {
            vidType v3 = y0n1y2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[13] += difference_num(n0y1y2, y3);
          }
          for(vidType idx3 = 0; idx3 < y0f0n1y2f1.size(); idx3++) {
            vidType v3 = y0f0n1y2f1.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[18] += intersection_num(n0f0y1y2, y3, v0);
          }
          for(vidType idx3 = 0; idx3 < y0y1n2.size(); idx3++) {
            vidType v3 = y0y1n2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[17] += intersection_num(y0y1n2, y3, v3);
            VertexSet n0n1n2y3; counter[10] += difference_num(difference_set(n0n1n2y3,difference_set(n0n1n2y3,y3, y0), y1), y2);
          }
          for(vidType idx3 = 0; idx3 < y0y1n2f2.size(); idx3++) {
            vidType v3 = y0y1n2f2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[16] += intersection_num(n0n1y2, y3);
            counter[6] += difference_num(y0y1n2f2, y3, v3);
          }
          for(vidType idx3 = 0; idx3 < n0n1y2.size(); idx3++) {
            vidType v3 = n0n1y2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            VertexSet n0n1n2y3; counter[9] += difference_num(difference_set(n0n1n2y3,difference_set(n0n1n2y3,y3, y0), y1), y2);
            counter[2] += difference_num(n0n1y2, y3, v3);
          }
          for(vidType idx3 = 0; idx3 < n0n1y2f0.size(); idx3++) {
            vidType v3 = n0n1y2f0.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[14] += intersection_num(n0f0n1y2, y3, v3);
          }
          for(vidType idx3 = 0; idx3 < y0n1n2.size(); idx3++) {
            vidType v3 = y0n1n2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[12] += intersection_num(n0y1n2, y3);
            counter[7] += difference_num(n0y1n2, y3);
          }
        }
        for(vidType idx2 = 0; idx2 < y0f0y1f1.size(); idx2++) {
          vidType v2 = y0f0y1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          VertexSet y0y1y2 = intersection_set(y0y1, y2);
          VertexSet y0f0y1f1y2f2 = intersection_set(y0f0y1f1, y2, v2);
          for(vidType idx3 = 0; idx3 < y0y1y2.size(); idx3++) {
            vidType v3 = y0y1y2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[19] += difference_num(y0y1y2, y3, v3);
            VertexSet n0n1n2y3; counter[15] += difference_num(difference_set(n0n1n2y3,difference_set(n0n1n2y3,y3, y0), y1), y2);
          }
          for(vidType idx3 = 0; idx3 < y0f0y1f1y2f2.size(); idx3++) {
            vidType v3 = y0f0y1f1y2f2.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[20] += intersection_num(y0f0y1f1y2f2, y3, v3);
          }
        }
        for(vidType idx2 = 0; idx2 < y0f0n1f1.size(); idx2++) {
          vidType v2 = y0f0n1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          VertexSet n0f0n1y2; difference_set(n0f0n1y2,difference_set(n0f0n1y2,y2, y0), y1);
          VertexSet n0y1n2f0 = difference_set(n0f0y1, y2, v0);
          for(vidType idx3 = 0; idx3 < n0y1n2f0.size(); idx3++) {
            vidType v3 = n0y1n2f0.begin()[idx3];
            VertexSet y3 = g.N(v3);
            counter[11] += intersection_num(n0f0n1y2, y3, v0);
          }
        }
      }
    }
  }
}

void automine_kmotif(Graph &g, unsigned k, std::vector<std::vector<uint64_t>> &counters) {
  std::cout << "Running AutoMine " << k << "-motif solver\n";
  if (k == 3) {
    automine_3motif(g, counters);
  } else if (k == 4) {
    automine_4motif(g, counters);
  } else if (k == 5) {
    automine_5motif(g, counters);
  } else {
    std::cout << "Not implemented yet\n";
    exit(0);
  }
}

