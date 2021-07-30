#pragma once
#include "operations.cuh"
#include "cutil_subset.h"

class cmap_gpu {
public:
  int nBuckets;   // # of buckets, must be 2^n
  int bucketSize; // size of each bucket
  int binSize;    // total size
  vidType *bins;
  cmap_gpu(unsigned nb, unsigned bz, unsigned n) :
    nBuckets(nb), bucketSize(bz), binSize(nb*bz) {
    auto bins_mem = n * nb * bz * sizeof(vidType);
    std::cout << "cmap memory allocation: " << bins_mem/1024/1024 << " MB\n";
    CUDA_SAFE_CALL(cudaMalloc((void**)&bins, bins_mem));
    CUDA_SAFE_CALL(cudaMemset(bins, 0, bins_mem));
  }
  inline __device__ void init_bin_counts(vidType *bin_counts) {
    unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);
    unsigned lid = threadIdx.x / WARP_SIZE; // local warp id, i.e. warp_lane
    auto offset = lid * nBuckets;
    for (auto i = thread_lane + offset; i < offset + nBuckets; i += WARP_SIZE)
      bin_counts[i] = 0;
    __syncwarp();
  }
  inline __device__ void insert(unsigned gid, vidType u, vidType *bin_counts) {
    unsigned lid = threadIdx.x / WARP_SIZE; // local warp id, i.e. warp_lane
    auto key = hash(u);
    auto index = atomicAdd(&bin_counts[get_offset(lid)+key], 1);
    bins[get_start(gid) + index * nBuckets + key] = u; // put into hash bin
    //bins[get_start(gid) + key*bucketSize + index] = u; // put into hash bin
  }
  inline __device__ bool lookup(unsigned gid, vidType v, vidType *bin_counts) {
    unsigned lid = threadIdx.x / WARP_SIZE;
    auto key = hash(v);
    auto len = bin_counts[key+get_offset(lid)];
    if (linear_search(v, bins, len, key+get_start(gid), nBuckets))
    //if (linear_search(v, bins, len, key*bucketSize+get_start(gid), 1))
      return true;
    return false;
  }
  inline __device__ void clean(vidType v, vidType *bin_counts) {
    unsigned lid = threadIdx.x / WARP_SIZE;
    auto key = hash(v);
    bin_counts[key + get_offset(lid)] = 0;
  }
private:
  inline __device__ vidType hash(vidType vid) { return mod<vidType>(vid); }
  inline __device__ vidType get_start(unsigned id) { return id * binSize; }
  inline __device__ vidType get_offset(unsigned id) { return id * nBuckets; }

  template <typename T>
  inline __device__ T mod(T key) { return key & (nBuckets-1); }
};

