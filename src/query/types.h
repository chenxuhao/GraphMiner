#pragma once
#include "common.h"

enum MatchingIndexType {
    VertexCentric = 0,
    EdgeCentric = 1
};

class TreeNode {
public:
  vidType id_;
  vidType parent_;
  int level_;
  int under_level_count_;
  int children_count_;
  int bn_count_;
  int fn_count_;
  vidType* under_level_;
  vidType* children_;
  vidType* bn_;
  vidType* fn_;
  size_t estimated_embeddings_num_;
public:
  TreeNode() {
    id_ = 0;
    under_level_ = NULL;
    bn_ = NULL;
    fn_ = NULL;
    children_ = NULL;
    parent_ = 0;
    level_ = 0;
    under_level_count_ = 0;
    children_count_ = 0;
    bn_count_ = 0;
    fn_count_ = 0;
    estimated_embeddings_num_ = 0;
  }
  ~TreeNode() {
    delete[] under_level_;
    delete[] bn_;
    delete[] fn_;
    delete[] children_;
  }
  void initialize(const int size) {
    under_level_ = new vidType[size];
    bn_ = new vidType[size];
    fn_ = new vidType[size];
    children_ = new vidType[size];
  }
};

// candidate search tree
typedef std::vector<TreeNode> CST;

class Edges {
public:
  int* offset_;
  int* edge_;
  int vertex_count_;
  int edge_count_;
  int max_degree_;
public:
  Edges() {
    offset_ = NULL;
    edge_ = NULL;
    vertex_count_ = 0;
    edge_count_ = 0;
    max_degree_ = 0;
  }
  ~Edges() {
    delete[] offset_;
    delete[] edge_;
  }
};

