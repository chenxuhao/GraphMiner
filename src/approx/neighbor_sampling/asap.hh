#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <random>
#include <math.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <set>
//#include <chrono>
//#include <atomic>
//#include <thread>
//#include <mutex>

struct Edge {
  Edge(uint32_t v_start_, uint32_t v_end_) : v_start(v_start_), v_end(v_end_) {
  }
  uint32_t v_start;
  uint32_t v_end;
};

/* Vertex structure
 * * (degree, edge_index to edge_list of its first neighbor edge)
 * */
struct Vertex {
  Vertex(uint32_t degree_, uint32_t edge_index_) : degree(degree_), edge_index(edge_index_) {
  }
  uint32_t degree;
  uint32_t edge_index;
};

class Subgraph {
  public:
    Subgraph() { }
    void insert_edge(Edge e) {
      total_edges++;
      edge_list.push_back(e);
      vertex_set.insert(e.v_start);
      vertex_set.insert(e.v_end);
    }
    uint32_t total_edges = 0;
    uint32_t total_vertices = 0;
    vector<struct Edge> edge_list;
    set<uint32_t> vertex_set;
};

inline double ASAP_chain_neighbor_sampler(Graph &graph, uint32_t chain_length, default_random_engine & rand_generator, uniform_int_distribution <uint32_t> &edge_distribution) {
  double prob_inverse = 0;
  Subgraph subgraph;
  map<uint32_t, Vertex> open_vertex_set;
  uint32_t last_edge_idx = edge_distribution(rand_generator);
  Edge e1(graph.get_src(last_edge_idx), graph.get_dst(last_edge_idx));
  subgraph.insert_edge(e1);
  open_vertex_set.emplace(e1.v_start, Vertex(graph.get_degree(e1.v_start), graph.edge_begin(e1.v_start)));
  open_vertex_set.emplace(e1.v_end, Vertex(graph.get_degree(e1.v_end), graph.edge_begin(e1.v_end)));
  prob_inverse = graph.E();
  uint32_t total_d_extending_edge = 0;
  for (uint32_t j = 0; j < chain_length - 2; j++) {
    total_d_extending_edge = 0;
    vector<uint32_t> open_set_neighbor;
    for (auto it = open_vertex_set.begin(); it != open_vertex_set.end(); it++) {
      uint32_t begin_idx = it->second.edge_index;
      uint32_t end_idx = it->second.edge_index + it->second.degree;
      for (uint32_t idx = begin_idx; idx < end_idx; idx++) {
        Edge edge(graph.get_src(idx), graph.get_dst(idx));
        if (subgraph.vertex_set.find(edge.v_end) != subgraph.vertex_set.end()) 
          continue;
        total_d_extending_edge++;
        open_set_neighbor.push_back(idx);
      }
    }
    prob_inverse *= total_d_extending_edge;
    if (total_d_extending_edge == 0) return 0;
    uniform_int_distribution <uint32_t> next_edge_distribution(0, total_d_extending_edge - 1);
    auto edge_idx = open_set_neighbor[next_edge_distribution(rand_generator)];
    Edge extending_edge(graph.get_src(edge_idx), graph.get_dst(edge_idx));
    subgraph.insert_edge(extending_edge);
    open_vertex_set.erase(extending_edge.v_start);
    open_vertex_set.emplace(extending_edge.v_end, Vertex(graph.get_degree(extending_edge.v_end), graph.edge_begin(extending_edge.v_end)));
  }
  return prob_inverse;
}

