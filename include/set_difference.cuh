#pragma once
#include "search.cuh"

// compute set difference: a - b
template <typename T = vidType>
__forceinline__ __device__ T difference_num_bs(T* a, T size_a, T* b, T size_b) {
  //if (size_a == 0) return 0;
  //assert(size_b != 0);
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    if (!binary_search(b, key, size_b))
      num += 1;
  }
  return num;
}

template <typename T = vidType>
__forceinline__ __device__ T difference_num_bs_cache(T* a, T size_a, T* b, T size_b) {
  //if (size_a == 0) return 0;
  //assert(size_b != 0);
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];
  __syncwarp();
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    if (!binary_search_2phase(b, cache, key, size_b))
      num += 1;
  }
  return num;
}

template <typename T = vidType>
__forceinline__ __device__ T difference_num(T* a, T size_a, T* b, T size_b) {
  return difference_num_bs_cache(a, size_a, b, size_b);
}

// compute set difference: a - b
template <typename T = vidType>
__forceinline__ __device__ T difference_num_bs(T* a, T size_a, T* b, T size_b, T upper_bound) {
  //if (size_a == 0) return 0;
  //assert(size_b != 0);
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && !binary_search(b, key, size_b))
      num += 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  return num;
}

template <typename T = vidType>
__forceinline__ __device__ T difference_num_bs_cache(T* a, T size_a, T* b, T size_b, T upper_bound) {
  //if (size_a == 0) return 0;
  //assert(size_b != 0);
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];
  __syncwarp();
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && !binary_search_2phase(b, cache, key, size_b))
      num += 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  return num;
}

template <typename T = vidType>
__forceinline__ __device__ T difference_num(T* a, T size_a, T* b, T size_b, T upper_bound) {
  return difference_num_bs_cache(a, size_a, b, size_b, upper_bound);
}

template <typename T = vidType>
__forceinline__ __device__ T difference_set_bs(T* a, T size_a, T* b, T size_b, T *c) {
  //if (size_a == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];

  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i]; // each thread picks a vertex as the key
    int found = 0;
    if (!binary_search(b, key, size_b))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}
 
template <typename T = vidType>
__forceinline__ __device__ T difference_set_bs_cache(T* a, T size_a, T* b, T size_b, T* c) {
  //if (size_a == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];
  __syncwarp();

  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i]; // each thread picks a vertex as the key
    int found = 0;
    if (!binary_search_2phase(b, cache, key, size_b))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}

template <typename T = vidType>
__forceinline__ __device__ T difference_set(T* a, T size_a, T* b, T size_b, T* c) {
  return difference_set_bs_cache(a, size_a, b, size_b, c);
}

// set difference: c = a - b
template <typename T = vidType>
__forceinline__ __device__ T difference_set_bs(T* a, T size_a, T* b, T size_b, T upper_bound, T* c) {
  //if (size_a == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];

  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i]; // each thread picks a vertex as the key
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && !binary_search(b, key, size_b))
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

// set difference: c = a - b
template <typename T = vidType>
__forceinline__ __device__ T difference_set_bs_cache(T* a, T size_a, T* b, T size_b, T upper_bound, T* c) {
  //if (size_a == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];
  __syncwarp();
  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i]; // each thread picks a vertex as the key
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && !binary_search_2phase(b, cache, key, size_b))
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
__forceinline__ __device__ T difference_set(T* a, T size_a, T* b, T size_b, T upper_bound, T* c) {
  return difference_set_bs_cache(a, size_a, b, size_b, upper_bound, c);
}

