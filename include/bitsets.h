
#pragma once
//#include <cuda.h>
//#include <assert.h>
#include "common.h"
#include "cutil_subset.h"

#ifdef USE_LONG
#define WIDTH 64
typedef uint64_t BSType;
typedef unsigned long long BSTypeGPU;
#else
#define WIDTH 32
typedef uint32_t BSType;
typedef unsigned BSTypeGPU;
#endif

class Bitsets {
  public:
    int num_sets;
    int num_bits_capacity;
    int num_bits;
    BSType** h_bit_vectors;
    BSType** d_bit_vectors;
    Bitsets() {}
    Bitsets(int n, int nbits) {
      alloc(n, nbits);
    }
    ~Bitsets() {}
    void set_size(int n, int nbits) {
      num_sets = n;
      num_bits_capacity = nbits;
      num_bits          = nbits;
    }
    void alloc(int n, int nbits) {
      num_sets = n;
      num_bits_capacity = nbits;
      num_bits          = nbits;
      h_bit_vectors = (BSType**)malloc(n * sizeof(BSType*));
      for (int i = 0; i < n; i++) {
        CUDA_SAFE_CALL(cudaMalloc(&h_bit_vectors[i], vec_size() * sizeof(BSType)));
        reset(i);
      }
      CUDA_SAFE_CALL(cudaMalloc(&d_bit_vectors, n * sizeof(BSType*)));
      CUDA_SAFE_CALL(cudaMemcpy(d_bit_vectors, h_bit_vectors, n * sizeof(BSType*), cudaMemcpyHostToDevice));
    }
    void clear() {
      for (int i = 0; i < num_sets; i++) reset(i);
      CUDA_SAFE_CALL(cudaMemcpy(d_bit_vectors, h_bit_vectors, num_sets * sizeof(BSType*), cudaMemcpyHostToDevice));
    }
    void clean() {
      for (int i = 0; i < num_sets; i++)
        if (h_bit_vectors[i] != NULL)
          cudaFree(h_bit_vectors[i]);
      if (d_bit_vectors != NULL)
        cudaFree(d_bit_vectors);
      if (h_bit_vectors != NULL)
        free(h_bit_vectors);
    }
    void reset(int i) {
      CUDA_SAFE_CALL(cudaMemset(h_bit_vectors[i], 0, vec_size() * sizeof(BSType)));
    }
    __device__ void set(int sid, int bid) {
      if (sid >= num_sets) printf("sid=%d, num_sets=%d\n", sid, num_sets);
      assert(sid < num_sets);
      assert(bid < num_bits);
      int bit_index = bid / WIDTH;
      BSTypeGPU bit_offset = 1;
      bit_offset <<= (bid % WIDTH);
      if ((d_bit_vectors[sid][bit_index] & bit_offset) == 0) { // test and set
        atomicOr((BSTypeGPU*)&d_bit_vectors[sid][bit_index], bit_offset);
      }
    }
    __device__ int count_num_ones(int sid, size_t bid) {
      return __popcll(d_bit_vectors[sid][bid]);
    }
    __device__ __host__ size_t vec_size() const {
      size_t bit_vector_size = (num_bits - 1) / WIDTH + 1;
      return bit_vector_size;
    }
}; 

template <typename T = uint32_t, int W = 32, int Z = 5>
class MultiBitsets {
  public:
    MultiBitsets() {}
    MultiBitsets(int m, int n, int nbits) {
      alloc(m, n, nbits);
    }
    ~MultiBitsets() {}
    void alloc(int m, int n, int nbits) {
      size = m;
      num_sets = n;
      num_bits = nbits;
      num_chunks = (num_bits - 1) / W + 1;
      set_size = size_t(n) * size_t(num_chunks);
      size_t mem_size = size_t(m) * set_size * sizeof(T);
      CUDA_SAFE_CALL(cudaMalloc(&begin, mem_size));
      CUDA_SAFE_CALL(cudaMemset(begin, 0, mem_size));
      std::cout << m << " bitsets of " << n << " sets, " << num_chunks << " chunks, " << nbits 
                << " bits, total memory size: " << float(mem_size)/(1024*1024) << " MB\n";
    }
    __device__ void warp_clear(int wid) {
      assert(wid<size);
      int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
      T* v = begin + wid * set_size;
      for (auto i = thread_lane; i < set_size; i += WARP_SIZE) {
        v[i] = 0;
      }
      __syncwarp();
    }
    __device__ void reset(int i) {
    }
    __device__ void warp_set(size_t offset, int sid, int bid, bool flag) {
      //assert(wid<size);
      //assert(sid<num_sets);
      //assert(bid<num_bits);
      T* v = begin + offset + sid * num_chunks;
      int thread_lane = threadIdx.x & (WARP_SIZE-1);
      unsigned active = __activemask();
      unsigned mask = __ballot_sync(active, flag);
      int cid = bid >> Z;
      //if (cid >= num_chunks) printf("wid=%d, tid=%d, sid=%d, bid=%d, cid=%d\n", wid, thread_lane, sid, bid, cid);
      //assert(cid<num_chunks);
      if (thread_lane == 0) v[cid] = mask;
    }
    __device__ bool get(size_t offset, int sid, int bid) {
      T* v = begin + offset + sid * num_chunks;
      int cid = bid >> Z;
      int x = bid & (W-1);
      return v[cid] & (1<<x);
    }
    __device__ int intersect_num(size_t offset, int nc, int sid1, int sid2) {
      T* v1 = begin + offset + sid1 * num_chunks;
      T* v2 = begin + offset + sid2 * num_chunks;
      int num = 0;
      int thread_lane = threadIdx.x & (WARP_SIZE-1);
      //for (int i = 0; i < nc; i++) {
      for (int i = thread_lane; i < nc; i+= WARP_SIZE) {
        num += __popc(v1[i] & v2[i]);
      }
      return num;
    }

  private:
    int size;
    int num_sets;
    int num_bits;
    int num_chunks;
    size_t set_size;
    T* begin;
};

