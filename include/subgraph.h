#include "common.h"
#include <bitset> 
class Subgraph : public std::vector<vidType> {
  private:
    std::bitset<16> connectivity; // TODO: max size is 6
  public:
    Subgraph(vidType v) { this->push_back(v); }
    Subgraph(Subgraph sg, vidType u) {
      for (auto v : sg) {
        if (u != v) this->push_back(v);
      }
      connectivity.reset();
      connectivity.set(0); // v0 and v1 are connected
    }
    vidType get_vertex(int i) { return (*this)[i]; }
    void pop() {
      int n = size();
      assert(n>2);
      auto start = (n-2)*(n-1)/2;
      for (int i = 0; i < n-1; i++)
        connectivity.set(start+i, 0);
      this->pop_back();
    }
    void push(vidType v, int parent, Graph &g) {
      this->push_back(v);
      int n = size();
      assert(n>2);
      auto start = (n-2)*(n-1)/2;
      connectivity.set(start+parent); // definitely connected with its parent
      for (int i = 0; i < n-1; i++) {
        if (i == parent) continue;
        if (g.is_connected(v, (*this)[i]))
          connectivity.set(start+i);
      }
    }
    bool is_connected_without(int idx) {
      assert(idx>=1);
      int n = size();
      for (int i = idx+1; i < n; i++) {
        auto start = i*(i-1)/2;
        bool has_one_edge = false;
        for (int j = 0; j < 3; j++) {
          if (j == idx) continue;
          if (connectivity.test(start+j)) {
            has_one_edge = true;
            break;
          }
        }
        if (!has_one_edge) return false;
      }
      return true;
    }
    inline int get_num(Graph &g, label_t a) {
      int n = 0;
      for (auto v : *this)
        if (a == g.get_vlabel(v)) n++;
      return n;
    }
    inline bool has_only_one(Graph &g, label_t a) {
      int n = 0;
      for (auto v : *this) {
        if (a == g.get_vlabel(v)) n++;
        if (n > 1) return false;
      }
      if (n == 1) return true;
      return false;
    }
    inline bool has_more_than_one(Graph &g, label_t a) {
      int n = 0;
      for (auto v : *this) {
        if (a == g.get_vlabel(v)) n++;
        if (n > 1) return true;
      }
      return false;
    }
    inline bool is_canonical(Graph &g, vidType v, int idx) {
      int n = size();
      // the new vertex id should be larger than the first vertex id
      if (v <= get_vertex(0)) return false;
      // the new vertex should not already exist in the subgragh
      for (int i = 1; i < n; ++i)
        if (v == get_vertex(i)) return false;
      // the new vertex should not already be extended by any previous vertex in the subgraph
      for (int i = 0; i < idx; ++i)
        if (g.is_connected(get_vertex(i), v)) return false;
      // the new vertex id should be larger than any vertex id after its source vertex in the subgraph
      for (int i = idx+1; i < n; ++i)
        if (v < get_vertex(i)) return false;
      return true;
    }
};

