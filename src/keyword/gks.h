// Copyright 2021 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#pragma once
#include "graph.h"
// Graph Keyword Search
// list the occurrances of patterns specified by keywords

class Keywords : public std::vector<label_t> {
  public:
    Keywords(label_t a, label_t b, label_t c) {
      this->push_back(a);
      this->push_back(b);
      this->push_back(c);
    }
    inline bool contains(label_t a) {
      for (auto label : *this)
        if (label == a) return true;
      return false;
    }
};

void GksSolver(Graph &g, int k, int nlabels, Keywords keywords, uint64_t &total, int n_devices, int chunk_size);
