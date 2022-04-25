#pragma once
typedef unsigned char SetType;

struct OrderedEdge {
  IndexT src;
  IndexT dst;
};

inline __device__ __host__ int get_init_pattern_id(BYTE src_label, BYTE dst_label, int nlabels) {
  return (int)src_label * nlabels + (int)dst_label;
}

inline __device__ __host__ int get_pattern_id(BYTE label0, BYTE label1, BYTE label2, int nlabels) {
  return nlabels * (nlabels * label0 + label1) + label2;
}

inline __device__ bool is_quick_automorphism(unsigned size, IndexT *vids, BYTE his2, BYTE his, IndexT src, IndexT dst) {
  if (dst <= vids[0]) return true;
  if (dst == vids[1]) return true;
  if (his == 0 && dst < vids[1]) return true;
  if (size == 2) {
  } else if (size == 3) {
    if (his == 0 && his2 == 0 && dst <= vids[2]) return true;
    if (his == 0 && his2 == 1 && dst == vids[2]) return true;
    if (his == 1 && his2 == 1 && dst <= vids[2]) return true;
  } else {
  }
  return false;
}

inline __device__ void swap(IndexT first, IndexT second) {
  if (first > second) {
    IndexT tmp = first;
    first = second;
    second = tmp;
  }
}

inline __device__ int compare(OrderedEdge oneEdge, OrderedEdge otherEdge) {
  swap(oneEdge.src, oneEdge.dst);
  swap(otherEdge.src, otherEdge.dst);
  if(oneEdge.src == otherEdge.src) return oneEdge.dst - otherEdge.dst;
  else return oneEdge.src - otherEdge.src;
}

inline __device__ bool is_edge_automorphism(unsigned size, IndexT *vids, BYTE *hiss, BYTE his, IndexT src, IndexT dst) {
  if (size < 3) return is_quick_automorphism(size, vids, hiss[2], his, src, dst);
  if (dst <= vids[0]) return true;
  if (his == 0 && dst <= vids[1]) return true;
  if (dst == vids[hiss[his]]) return true;
  OrderedEdge added_edge;
  added_edge.src = src;
  added_edge.dst = dst;
  for (unsigned index = his + 1; index < size; ++index) {
    OrderedEdge edge;
    edge.src = vids[hiss[index]];
    edge.dst = vids[index];
    int cmp = compare(added_edge, edge);
    if(cmp <= 0) return true;
  }
  return false;
}

