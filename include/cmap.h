#pragma once

#include <algorithm>
#include "defines.h"

// ccode map data structure
#if defined(IDENT_CMAP)
template <typename KT=uint32_t, typename VT=uint8_t>
class cmap_t {
private:
  std::vector<VT> cmap; // ccode map
public:
  cmap_t() {}
  cmap_t(size_t n) { init(n); }
  ~cmap_t() {}
  void init(size_t n) {
    cmap.resize(n);
    std::fill(cmap.begin(), cmap.end(), 0);
  }
  VT get(KT k) { return cmap[k]; }
  void set(KT k, VT v) { cmap[k] = v; }
  void clear() { std::fill(cmap.begin(), cmap.end(), 0); }
};

#elif defined(STL_CMAP)
#include <boost/unordered_map.hpp>
#include "robin_hood.h"
#include <unordered_map>

template <typename KT=uint32_t, typename VT=uint8_t>
class cmap_t {
private:
  //galois::gstl::UnorderedMap<KT, VT> cmap; // ccode map
  //boost::unordered_map<KT, VT> cmap; // ccode map
  std::unordered_map<KT, VT> cmap; // ccode map
  //robin_hood::unordered_map<KT, VT> cmap; // ccode map
public:
  cmap_t() {}
  cmap_t(size_t n) { init(n); }
  ~cmap_t() {}
  void init(size_t n) {
    cmap.reserve(n);
  }
  VT get(KT k) const {
    auto it = cmap.find(k);
    if (it == cmap.end()) return 0; 
    return it->second;
  }
  void set(KT k, VT v) {
    cmap[k] = v;
  }
  void clear() { }
};

#elif defined(CUCKOO_CMAP)
#include "cuckoo.h"

template <typename KT=uint32_t, typename VT=uint8_t>
using cmap_t = cuckoo_map<KT, VT, CUCKOO_WAY>;

#endif

template <typename KT=uint32_t, typename VT=uint8_t>
static inline bool is_clique(uint32_t level, vidType v, cmap_t<KT,VT> &cmap) {
  return cmap.get(KT(v)) == VT(level);
}

typedef cmap_t<uint32_t,uint8_t> cmap8_t;
typedef cmap_t<uint32_t,uint16_t> cmap16_t;
