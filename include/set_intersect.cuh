#pragma once
#include "search.cuh"

// warp-wise intersetion of two lists using the binary seach algorithm
template <typename T = vidType>
__forceinline__ __device__ T intersect_bs(T* a, T size_a, T* b, T size_b, T* c) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];
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
  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    int found = 0;
    vidType key = lookup[i]; // each thread picks a vertex as the key
    if (binary_search(search, key, search_size))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}

// warp-wise intersetion of two lists using the 2-phase binary seach algorithm with caching
template <typename T = vidType>
__forceinline__ __device__ T intersect_bs_cache(T* a, T size_a, T* b, T size_b, T* c, T* cache) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];
  vidType *lookup = a;
  vidType *search = b;
  vidType lookup_size = size_a;
  vidType search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  if (thread_lane == 0) count[warp_lane] = 0;
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    vidType key = lookup[i]; // each thread picks a vertex as the key
    int found = 0;
    if (binary_search_2phase(search, cache, key, search_size))
      found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}

template <typename T = vidType>
__forceinline__ __device__ T intersect_bs_cache(T* a, T size_a, T* b, T size_b, T* c) {
  //if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];
  __shared__ T cache[BLOCK_SIZE];
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
  if (thread_lane == 0) count[warp_lane] = 0;
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    vidType key = lookup[i]; // each thread picks a vertex as the key
    int found = 0;
    if (binary_search_2phase(search, cache, key, search_size))
      found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}

// warp-wise intersection using hybrid method (binary search + merge)
template <typename T = vidType>
__forceinline__ __device__ T intersect(T* a, T size_a, T *b, T size_b, T* c) {
  //if (size_a > ADJ_SIZE_THREASHOLD && size_b > ADJ_SIZE_THREASHOLD)
  //  return intersect_merge(a, size_a, b, size_b, c);
  //else
    return intersect_bs_cache(a, size_a, b, size_b, c);
}

template <typename T = vidType>
__forceinline__ __device__ T intersect_bs(T* a, T size_a, T* b, T size_b, T upper_bound, T* c) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];
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
  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    vidType key = lookup[i]; // each thread picks a vertex as the key
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && binary_search(search, key, search_size))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
    mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  return count[warp_lane];
}

template <typename T = vidType>
__forceinline__ __device__ T intersect_bs_cache(T* a, T size_a, T* b, T size_b, T upper_bound, T* c) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];
  __shared__ T cache[BLOCK_SIZE];
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
  if (thread_lane == 0) count[warp_lane] = 0;
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    vidType key = lookup[i]; // each thread picks a vertex as the key
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && binary_search_2phase(search, cache, key, search_size))
      found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
    mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  return count[warp_lane];
}

// warp-wise intersection using hybrid method (binary search + merge)
template <typename T = vidType>
__forceinline__ __device__ T intersect(T* a, T size_a, T *b, T size_b, T upper_bound, T* c) {
  return intersect_bs_cache(a, size_a, b, size_b, upper_bound, c);
}

// warp-wise intersection using binary search
template <typename T = vidType>
__forceinline__ __device__ T intersect_num_bs(T* a, T size_a, T* b, T size_b) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  T num = 0;
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
  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i];
    if (binary_search(search, key, search_size))
      num += 1;
  }
  return num;
}

// warp-wise intersection using 2-phase binary search with caching
template <typename T = vidType>
__forceinline__ __device__ T intersect_num_bs_cache(T* a, T size_a, T* b, T size_b, T* cache) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  T num = 0;
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
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i]; // each thread picks a vertex as the key
    if (binary_search_2phase(search, cache, key, search_size))
      num += 1;
  }
  return num;
}
/*
template <typename T = vidType>
__forceinline__ __device__ T intersect_num_bs_cache(T* a, T a_size, T* b, T b_size) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  __shared__ T cache[BLOCK_SIZE];
  T num = 0;
  if (a_size > b_size) {
    cache[threadIdx.x] = a[thread_lane * a_size / WARP_SIZE];
    __syncwarp();
    for (auto i = thread_lane; i < b_size; i += WARP_SIZE) {
      auto key = b[i]; // each thread picks a vertex as the key
      if (binary_search_2phase(a, cache, key, a_size))
        num += 1;
    }
  } else {
    cache[threadIdx.x] = b[thread_lane * b_size / WARP_SIZE];
    __syncwarp();
    for (auto i = thread_lane; i < a_size; i += WARP_SIZE) {
      auto key = a[i]; // each thread picks a vertex as the key
      if (binary_search_2phase(b, cache, key, b_size))
        num += 1;
    }
  }
  return num;
}
*/
template <typename T = vidType>
__forceinline__ __device__ T intersect_num_bs_cache(T* a, T size_a, T* b, T size_b) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T cache[BLOCK_SIZE];
  T num = 0;
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
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i]; // each thread picks a vertex as the key
    if (binary_search_2phase(search, cache, key, search_size))
      num += 1;
  }
  return num;
}

// warp-wise intersection using hybrid method (binary search + merge)
template <typename T = vidType>
__forceinline__ __device__ T intersect_num(T* a, T size_a, T *b, T size_b) {
  //if (size_a > ADJ_SIZE_THREASHOLD && size_b > ADJ_SIZE_THREASHOLD)
  //  return intersect_num_merge(a, size_a, b, size_b);
  //else
    return intersect_num_bs_cache(a, size_a, b, size_b);
}

// warp-wise intersection with upper bound using binary search
template <typename T = vidType>
__forceinline__ __device__ T intersect_num_bs(T *a, T size_a, T *b, T size_b, T upper_bound) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  T num = 0;
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
  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && binary_search(search, key, search_size))
      found = 1;
    num += found;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  __syncwarp();
  return num;
}

// warp-wise intersection with upper bound using 2-phase binary search with caching
template <typename T = vidType>
__forceinline__ __device__ T intersect_num_bs_cache(T* a, T size_a, T* b, T size_b, T upper_bound) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  T num = 0;
  T *lookup = a;
  T *search = b;
  T lookup_size = size_a;
  T search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i]; // each thread picks a vertex as the key
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && binary_search_2phase(search, cache, key, search_size))
      found = 1;
    num += found;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  __syncwarp();
  return num;
}

// warp-wise intersection using hybrid method (binary search + merge)
template <typename T = vidType>
__forceinline__ __device__ T intersect_num(T* a, T size_a, T *b, T size_b, T upper_bound) {
  //if (size_a > ADJ_SIZE_THREASHOLD && size_b > ADJ_SIZE_THREASHOLD)
  //  return intersect_num_merge(a, size_a, b, size_b, upper_bound);
  //else
    return intersect_num_bs_cache(a, size_a, b, size_b, upper_bound);
}

template <typename T = vidType>
__forceinline__ __device__ T intersect_num(T* a, T size_a, T* b, T size_b, T upper_bound, T ancestor) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  AccType num = 0;
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
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i];
    unsigned active = __activemask();
    __syncwarp(active);
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (key != ancestor && is_smaller && binary_search_2phase(search, cache, key, search_size))
      found = 1;
    num += found;
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  return num;
}

template <typename T = vidType>
__forceinline__ __device__ T intersect_num(T* a, T size_a, T* b, T size_b, T* ancestor, int n) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  AccType num = 0;
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
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i];
    bool valid = true;
    for (int j = 0; j < n; j++) {
      if (key == ancestor[j]) {
        valid = false;
        break;
      }
    }
    if (valid && binary_search_2phase(search, cache, key, search_size))
      num += 1;
  }
  return num;
}

