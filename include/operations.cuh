#pragma once
#include "cutil_subset.h"
#include "set_intersect.cuh"
#include "set_difference.cuh"

template <typename T>
__forceinline__ __device__ T warp_reduce(T val) {
  T sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8);
  sum += SHFL_DOWN(sum, 4);
  sum += SHFL_DOWN(sum, 2);
  sum += SHFL_DOWN(sum, 1);
  sum  = SHFL(sum, 0);
  return sum;
}

__forceinline__ __device__ void warp_reduce_iterative(vidType &val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  val  = SHFL(val, 0);
}

// from http://forums.nvidia.com/index.php?showtopic=186669
static __device__ unsigned get_smid(void) {
  unsigned ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

#define VLIST_CACHE_SIZE 256
__forceinline__ __device__ void warp_load_mem_to_shm(vidType* from, vidType* to, vidType len) {
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  for (vidType id = thread_lane; id < len; id += WARP_SIZE) {
    to[id] = from[id];
  }
  __syncwarp();
}

__forceinline__ __device__ int list_smaller(vidType bound, vidType *in, vidType size_in, vidType *out) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  __shared__ int count[WARPS_PER_BLOCK];
  if (thread_lane == 0) count[warp_lane] = 0;
  __syncwarp();
  for (auto i = thread_lane; i < size_in; i += WARP_SIZE) {
    int found = 0;
    if (in[i] < bound) found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) out[count[warp_lane]+idx-1] = in[i];
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
    __syncwarp(active);
    if (mask != active) break;
  }
  __syncwarp();
  return count[warp_lane];
}

__forceinline__ __device__ unsigned count_smaller(vidType bound, vidType *a, vidType size_a) {
  if (size_a == 0) return 0;
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  __shared__ unsigned count[WARPS_PER_BLOCK];
  __shared__ unsigned begin[WARPS_PER_BLOCK];
  __shared__ unsigned end[WARPS_PER_BLOCK];
  if (thread_lane == 0) {
    count[warp_lane] = 0;
    begin[warp_lane] = 0;
    end[warp_lane] = size_a;
  }
  __syncwarp();
  bool found = false;
  int mid = 0;
  int l = begin[warp_lane];
  int r = end[warp_lane];
  while (r-l > 32*4) {
    mid = l + (r - l) / 2; 
    auto value = a[mid];
    if (value == bound) {
      found = true;
      break;
    }
    if (thread_lane == 0) {
    if (value < bound) begin[warp_lane] = mid + 1;
    else end[warp_lane] = mid - 1;
    }
    __syncwarp();
    l = begin[warp_lane];
    r = end[warp_lane];
  }
  if (found) return mid;
  if (thread_lane == 0) count[warp_lane] = begin[warp_lane];
  for (auto i = thread_lane + l; i < r; i += WARP_SIZE) {
    int found = 0;
    if (a[i] < bound) found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
    __syncwarp(active);
    if (mask != FULL_MASK) break;
  }
  return count[warp_lane];
}

template <typename T = vidType>
__forceinline__ __device__ T all_pairs_compare(T a, T b) {
  T count = 0;
  for (int i = 0; i < WARP_SIZE; i ++)
    count += a == SHFL(b, i) ? 1 : 0;
  return count;
}

template <typename T = vidType>
__forceinline__ __device__ T all_pairs_compare(T key, T* tile) {
  T count = 0;
  for (int i = 0; i < WARP_SIZE; i ++)
    count += key == tile[i] ? 1 : 0;
  return count;
}

template <typename T = vidType>
__forceinline__ __device__ void all_pairs_compare(T key, T* b, T *pos, T* c) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  bool found = false;
  for (int i = 0; i < WARP_SIZE; i ++) {
    if (key == b[i]) {
      found = true;
      break;
    }
  }
  unsigned active = __activemask();
  unsigned mask = __ballot_sync(active, found);
  auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
  if (found) c[pos[0]+idx-1] = key;
  if (thread_lane == 0) pos[0] += __popc(mask);
}

// warp-wise intersetion of two lists using the merge-based algorithm with blocking
template <typename T = vidType>
__forceinline__ __device__ T intersect_merge(T* a, T size_a, T* b, T size_b, T* c) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the cta
  int p = warp_lane * WARP_SIZE;
  __shared__ T count[WARPS_PER_BLOCK];
  __shared__ T cache[BLOCK_SIZE];
  if (thread_lane == 0) count[warp_lane] = 0;
  auto a_off = thread_lane; 
  auto b_off = thread_lane; 
  T key_a = -1;
  if (a_off < size_a) key_a = a[a_off];
  if (b_off < size_b) cache[threadIdx.x] = b[b_off];
  else cache[threadIdx.x] = -2;
  __syncwarp();
  while (1) {
    T last_a = SHFL(key_a, WARP_SIZE-1);
    T last_b = cache[p+WARP_SIZE-1];
    int found = 0;
    if (key_a >= 0 && binary_search_enhanced(&cache[p], key_a, WARP_SIZE))
      found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key_a;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
    bool done_a = last_a<0;
    bool done_b = last_b<0;
    __syncwarp();
    if (done_a && done_b) break;
    if (done_a) last_a = a[size_a-1];
    if (done_b) last_b = b[size_b-1];
    if (last_a == last_b) {
      if (done_a || done_b) break;
      a_off += WARP_SIZE;
      b_off += WARP_SIZE;
      if (a_off < size_a) key_a = a[a_off];
      else key_a = -1;
      if (b_off < size_b) cache[threadIdx.x] = b[b_off];
      else cache[threadIdx.x] = -2;
    } else if (last_a > last_b) {
      if (done_b) break;
      b_off += WARP_SIZE;
      if (b_off < size_b) cache[threadIdx.x] = b[b_off];
      else cache[threadIdx.x] = -2;
    } else {
      if (done_a) break;
      a_off += WARP_SIZE;
      if (a_off < size_a) key_a = a[a_off];
      else key_a = -1;
    }
    __syncwarp();
  }
  return count[warp_lane];
}

// warp-wise intersetion of two lists using the merge based algorithm
template <typename T = vidType>
__forceinline__ __device__ T intersect_num_hybrid(T* a, T size_a, T* b, T size_b) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the cta
  __shared__ T cache[BLOCK_SIZE];           // cache for 2-phase binary search
  int p = warp_lane * WARP_SIZE;
  T count = 0;
  if (size_a > ADJ_SIZE_THREASHOLD && size_b > ADJ_SIZE_THREASHOLD) {
    auto a_off = thread_lane; 
    auto b_off = thread_lane; 
    while (1) {
      T key_a = -1;
      if (a_off < size_a) key_a = a[a_off];
      if (b_off < size_b) cache[threadIdx.x] = b[b_off];
      else cache[threadIdx.x] = -2;
      __syncwarp();
      T last_a = SHFL(key_a, WARP_SIZE-1);
      T last_b = cache[p+WARP_SIZE-1];
      if (a_off < size_a) {
        //for (int i = 0; i < WARP_SIZE; i ++)
          //count += key_a == cache[p+i] ? 1 : 0;
        int l = 0;
        int r = WARP_SIZE-1;
        bool found = false;
        while (r >= l) { 
          int mid = l + (r - l) / 2; 
          auto val = cache[p+mid];
          if (val == key_a) { found = true; break; }
          if (val < key_a && val >= 0) l = mid + 1;
          else r = mid - 1;
        }
        if (found) count += 1;
      }
      bool done_a = last_a<0;
      bool done_b = last_b<0;
      if (done_a && done_b) break;
      if (done_a) last_a = a[size_a-1];
      if (done_b) last_b = b[size_b-1];
      if (last_a == last_b) {
        if (done_a || done_b) break;
        a_off += WARP_SIZE;
        b_off += WARP_SIZE;
      } else if (last_a > last_b) {
        if (done_b) break;
        b_off += WARP_SIZE;
      } else {
        if (done_a) break;
        a_off += WARP_SIZE;
      }
    }
  } else { // binary search
    T* lookup = a;
    T* search = b;
    T lookup_size = size_a;
    T search_size = size_b;
    if (size_a > size_b) {
      lookup = b;
      search = a;
      lookup_size = size_b;
      search_size = size_a;
    }
    cache[threadIdx.x] = search[thread_lane * search_size / WARP_SIZE];
    __syncwarp();
    for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
      auto key = lookup[i]; // each thread picks a vertex as the key
      bool found = false;
      // phase 1: search in the cache
      int mid = 0;
      int bottom = 0;
      int top = WARP_SIZE;
      while (top > bottom + 1) {
        mid = (top + bottom) / 2;
        T y = cache[p + mid];
        if (key == y) { found = true; break; }
        if (key < y) top = mid;
        if (key > y) bottom = mid;
      }
      if (!found) {
        //phase 2: search in global memory
        bottom = bottom * search_size / WARP_SIZE;
        top = top * search_size / WARP_SIZE - 1;
        while (top >= bottom) {
          mid = (top + bottom) / 2;
          T y = search[mid];
          if (key == y) { found = true; break; }
          if (key < y) top = mid - 1;
          else bottom = mid + 1;
        }
      }
      if (found)  count += 1;
    }
  }
  return count;
}

// set intersection using warp-based HINDEX
template <typename T = vidType>
__forceinline__ __device__ T intersect_warp_hindex(T *a, T size_a, T *b, T size_b, T* bins, T* bin_counts) {
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
 
  int binSize = NUM_BUCKETS * BUCKET_SIZE;
  int binStart = warp_id * binSize;
  int binOffset = warp_lane * NUM_BUCKETS;

  auto lookup = a;
  auto search = b;
  auto lookup_size = size_a;
  auto search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  //__shared__ vidType max_vid[WARPS_PER_BLOCK];
  //max_vid[warp_lane] = lookup[lookup_size-1];

  // hash the shorter set
  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto vid = lookup[i];
    T key = vid % NUM_BUCKETS; // hash
    T index = atomicAdd(&bin_counts[key + binOffset], 1);
    bins[index * NUM_BUCKETS + key + binStart] = vid; // put into hash bin
  }
  __syncwarp();

  T count = 0;
  // probe the larger set
  for (auto i = thread_lane; i < search_size; i += WARP_SIZE) {
    auto vid = search[i];
    //int is_smaller = vid <= max_vid[warp_lane] ? 1 : 0;
    //if (is_smaller) {
    T key = vid % NUM_BUCKETS; // hash key
    auto len = bin_counts[key + binOffset];
    count += linear_search(vid, bins, len, key+binStart, NUM_BUCKETS);
    //}
    //unsigned active = __activemask();
    //unsigned mask = __ballot_sync(active, is_smaller);
    //if (mask != FULL_MASK) break;
  }
  __syncwarp();
  return count;
}

