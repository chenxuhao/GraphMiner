// This code is modified from AutoMine and GraphZero
// Daniel Mawhirter and Bo Wu. SOSP 2019.
// AutoMine: Harmonizing High-Level Abstraction and High Performance for Graph Mining

#include "VertexSet.h"
#include <cassert>

// VertexSet static members
thread_local std::vector<vidType*> VertexSet::buffers_exist(0);
thread_local std::vector<vidType*> VertexSet::buffers_avail(0);
vidType VertexSet::MAX_DEGREE = 1;

void VertexSet::release_buffers() {
  buffers_avail.clear();
  while(buffers_exist.size() > 0) {
    delete[] buffers_exist.back();
    buffers_exist.pop_back();
  }
}

vidType VertexSet::difference_buf(vidType *outBuf, const VertexSet &other) const {
  vidType idx_l = 0, idx_r = 0;
  vidType return_set_size = 0;
  while(idx_l < set_size && idx_r < other.set_size) {
    vidType left = ptr[idx_l];
    vidType right = other.ptr[idx_r];
    if(left <= right) idx_l++;
    if(right <= left) idx_r++;
    if(left < right && left != other.vid) {
      assert(idx_l>return_set_size);
      outBuf[return_set_size++] = left;
    }
  }
  while(idx_l < set_size) {
    vidType left = ptr[idx_l];
    idx_l++;
    if(left != other.vid) {
      assert(idx_l>return_set_size);
      outBuf[return_set_size++] = left;
    }
  }
  return return_set_size;
}

//outBuf may be the same as this->ptr
vidType VertexSet::difference_buf(vidType *outBuf, const VertexSet &other, vidType upper) const {
  vidType idx_l = 0, idx_r = 0;
  vidType return_set_size = 0;
  while(idx_l < set_size && idx_r < other.set_size) {
    vidType left = ptr[idx_l];
    vidType right = other.ptr[idx_r];
    if(left >= upper) break;
    if(right >= upper) break;
    if(left <= right) idx_l++;
    if(right <= left) idx_r++;
    if(left < right && left != other.vid) outBuf[return_set_size++] = left;
  }
  while(idx_l < set_size) {
    vidType left = ptr[idx_l];
    if(left >= upper) break;
    idx_l++;
    if(left != other.vid) {
      outBuf[return_set_size++] = left;
    }
  }
  return return_set_size;
}

vidType VertexSet::difference_ns(const VertexSet &other, vidType upper) const {
  vidType idx_l = 0, idx_r = 0, idx_out = 0;
  while(idx_l < set_size && idx_r < other.set_size) {
    vidType left = ptr[idx_l];
    vidType right = other.ptr[idx_r];
    if(left >= upper) break;
    if(right >= upper) break;
    if(left <= right) idx_l++;
    if(right <= left) idx_r++;
    if(left < right && left != other.vid) idx_out++;
  }
  while(idx_l < set_size) {
    vidType left = ptr[idx_l];
    if(left >= upper) break;
    idx_l++;
    if(left != other.vid) {
      idx_out++;
    }
  }
  return idx_out;
}

