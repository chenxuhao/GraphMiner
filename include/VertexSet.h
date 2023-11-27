#pragma once
#include "defines.h"
#include "common.h"
#include "timer.h"
#include "custom_alloc.h"
constexpr vidType VID_MIN = 0;
constexpr vidType VID_MAX = std::numeric_limits<vidType>::max();

inline vidType bs(vidType* ptr, vidType set_size, vidType o){
  vidType idx_l = -1;
  vidType idx_r = set_size;
  //guarantees in this area is that idx_l is before where we would put o 
  while(idx_r-idx_l > 1){
    vidType idx_t = (idx_l+idx_r)/2;
    if(ptr[idx_t] < o)idx_l = idx_t;
    else idx_r = idx_t;
  }
  return idx_l+1;
}

class VertexSet {
private: // memory managed regions for per-thread intermediates
  static thread_local std::vector<vidType*> buffers_exist, buffers_avail;
public:
  static void release_buffers();
  static vidType MAX_DEGREE;
  vidType *ptr;
private:
  vidType set_size, vid;
  const bool pooled;
public:
  VertexSet() : set_size(0), vid(-1), pooled(true) {
    if(buffers_avail.size() == 0) { 
      vidType *p = custom_alloc_local<vidType>(MAX_DEGREE);
      buffers_exist.push_back(p);
      buffers_avail.push_back(p);
    }
    ptr = buffers_avail.back();
    buffers_avail.pop_back();
  }
  VertexSet(vidType *p, vidType s, vidType id) : 
    ptr(p), set_size(s), vid(id), pooled(false) {}
  VertexSet(const VertexSet&)=delete;
  VertexSet& operator=(const VertexSet&)=delete;
  VertexSet(VertexSet&&)=default;
  VertexSet& operator=(VertexSet&&)=default;
  ~VertexSet() {
    if(pooled) {
      buffers_avail.push_back(ptr);
    }
  }
  vidType size() const { return set_size; }
  VertexSet operator &(const VertexSet &other) const {
    VertexSet out;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) out.ptr[out.set_size++] = left;
    }
    return out;
  }
  uint32_t get_intersect_num(const VertexSet &other) const {
    uint32_t num = 0;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) num++;
    }
    return num;
  }

  void print() const {
    std::copy(ptr, ptr+set_size, std::ostream_iterator<vidType>(std::cout, " "));
  }

  vidType difference_buf(vidType *outBuf, const VertexSet &other) const;

  VertexSet operator -(const VertexSet &other) const {
    VertexSet out;
    out.set_size = difference_buf(out.ptr, other); 
    return out;
  }

  VertexSet& difference(VertexSet& dst, const VertexSet &other) const {
    dst.set_size = difference_buf(dst.ptr, other);
    return dst;
  }

  VertexSet intersect(const VertexSet &other, vidType upper) const {
    VertexSet out;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) out.ptr[out.set_size++] = left;
    }
    return out;
  }

  vidType intersect_ns(const VertexSet &other, vidType upper) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) idx_out++;
    }
    return idx_out;
  }

  VertexSet intersect_except(const VertexSet &other, vidType ancestor) const {
    VertexSet out;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right && left != ancestor) out.ptr[out.set_size++] = left;
    }
    return out;
  }

  VertexSet intersect_bound_except(const VertexSet &other, vidType upper, vidType ancestor) const {
    VertexSet out;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right && left != ancestor) out.ptr[out.set_size++] = left;
    }
    return out;
  }

  vidType intersect_ns_bound_except(const VertexSet &other, vidType upper, vidType ancestor) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right && left != ancestor) idx_out++;
    }
    return idx_out;
  }

  vidType intersect_ns_except(const VertexSet &other, vidType ancestor) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right && left != ancestor) idx_out++;
    }
    return idx_out;
  }

  vidType intersect_ns_except(const VertexSet &other, vidType ancestorA, vidType ancestorB) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right && left != ancestorA && left != ancestorB) idx_out++;
    }
    return idx_out;
  }

  vidType intersect_ns_except(const VertexSet &other, VertexList &nodes) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) {
        bool found = false;
        for (auto w : nodes) if (left == w) { found = true; break; }
        if (!found) idx_out++;
      }
    }
    return idx_out;
  }

  vidType intersect_ns_bound_except(const VertexSet &other, vidType upper, VertexList &nodes) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) {
        bool found = false;
        for (auto w : nodes) if (left == w) { found = true; break; }
        if (!found) idx_out++;
      }
    }
    return idx_out;
  }

  //outBuf may be the same as this->ptr
  vidType difference_buf(vidType *outBuf, const VertexSet &other, vidType upper) const;

  VertexSet difference(const VertexSet &other, vidType upper) const {
    VertexSet out;
    out.set_size = difference_buf(out.ptr, other, upper);
    return out;
  }

  VertexSet& difference(VertexSet& dst, const VertexSet &other, vidType upper) const {
    dst.set_size = difference_buf(dst.ptr, other, upper);
    return dst;
  }

  vidType difference_ns(const VertexSet &other, vidType upper) const;

  VertexSet bounded(vidType up) const {
    if(set_size > 64) {
      vidType idx_l = -1;
      vidType idx_r = set_size;
      while(idx_r-idx_l > 1) {
        vidType idx_t = (idx_l+idx_r)/2;
        if(ptr[idx_t] < up)idx_l = idx_t;
        else idx_r = idx_t;
      }
      return VertexSet(ptr,idx_l+1,vid);
    } else {
      vidType idx_l = 0;
      while(idx_l < set_size && ptr[idx_l] < up) ++idx_l;
      return VertexSet(ptr,idx_l,vid);
    }
  }
  vidType *data() const { return ptr; }
  const vidType *begin() const { return ptr; }
  const vidType *end() const { return ptr+set_size; }
  void add(vidType v) { ptr[set_size++] = v; }
  void clear() { set_size = 0; }
  vidType& operator[](size_t i) { return ptr[i]; }
  const vidType& operator[](size_t i) const { return ptr[i]; }
};

inline VertexSet difference_set(const VertexSet& a, const VertexSet& b) {
  return a-b;
}

inline VertexSet& difference_set(VertexSet& dst, const VertexSet& a, const VertexSet& b) {
  return a.difference(dst, b);
}

inline VertexSet difference_set(const VertexSet& a, const VertexSet& b, vidType up) {
  return a.difference(b,up);
}

inline VertexSet& difference_set(VertexSet& dst, const VertexSet& a, const VertexSet& b, vidType up) {
  return a.difference(dst, b,up);
}

inline uint64_t difference_num(const VertexSet& a, const VertexSet& b) {
  return (a-b).size();
}

inline uint64_t difference_num(const VertexSet& a, const VertexSet& b, vidType up) {
  return a.difference_ns(b,up);
}

inline VertexSet intersection_set(const VertexSet& a, const VertexSet& b) {
  return a & b;
}

inline VertexSet intersection_set(const VertexSet& a, const VertexSet& b, vidType up) {
  return a.intersect(b,up);
}

inline uint64_t intersection_num(const VertexSet& a, const VertexSet& b) {
  return a.get_intersect_num(b);
}

inline uint64_t intersection_num(const VertexSet& a, const VertexSet& b, vidType up) {
  return a.intersect_ns(b, up);
}

inline VertexSet intersection_set_except(const VertexSet& a, const VertexSet& b, vidType ancestor) {
  return a.intersect_except(b, ancestor);
}

inline uint64_t intersection_num_except(const VertexSet& a, const VertexSet& b, vidType ancestor) {
  return a.intersect_ns_except(b, ancestor);
}

inline uint64_t intersection_num_except(const VertexSet& a, const VertexSet& b, vidType ancestorA, vidType ancestorB) {
  return a.intersect_ns_except(b, ancestorA, ancestorB);
}

inline uint64_t intersection_num_except(const VertexSet& a, const VertexSet& b, VertexList &nodes) {
  return a.intersect_ns_except(b, nodes);
}

inline uint64_t intersection_num_bound_except(const VertexSet& a, const VertexSet& b, vidType up, vidType ancestor) {
  return a.intersect_ns_bound_except(b, up, ancestor);
}

inline uint64_t intersection_num_bound_except(const VertexSet& a, const VertexSet& b, vidType up, VertexList &nodes) {
  return a.intersect_ns_bound_except(b, up, nodes);
}

inline VertexSet bounded(const VertexSet&a, vidType up) {
  return a.bounded(up);
}

inline void intersection(const VertexSet &a, const VertexSet &b, VertexSet &c) {
  vidType idx_l = 0, idx_r = 0;
  while(idx_l < a.size() && idx_r < b.size()) {
    vidType left = a[idx_l];
    vidType right = b[idx_r];
    if(left <= right) idx_l++;
    if(right <= left) idx_r++;
    if(left == right) c.add(left);
  }
}

