#pragma once
#include "graph.h"
#include "operations.cuh"
#include "cutil_subset.h"

class GraphGPU {
protected:
  vidType num_vertices;
  eidType num_edges;
  int device_id, n_gpu; // no. of GPUs
  eidType *d_rowptr; // row pointers
  vidType *d_colidx; // column induces
  vidType *d_src_list, *d_dst_list; // for COO format
  vlabel_t *d_labels;
public:
  GraphGPU() : device_id(0), n_gpu(1) {}
  GraphGPU(Graph &g) : device_id(0), n_gpu(1) { init(g); }
  GraphGPU(Graph &g, int n, int m) : device_id(n), n_gpu(m) { init(g); }
  inline __device__ __host__ bool valid_vertex(vidType vertex) { return (vertex < num_vertices); }
  inline __device__ __host__ bool valid_edge(eidType edge) { return (edge < num_edges); }
  inline __device__ __host__ vidType get_src(eidType eid) const { return d_src_list[eid]; }
  inline __device__ __host__ vidType get_dst(eidType eid) const { return d_dst_list[eid]; }
  inline __device__ __host__ vidType* N(vidType vid) { return d_colidx + d_rowptr[vid]; }
  inline __device__ __host__ eidType getOutDegree(vidType src) { return d_rowptr[src+1] - d_rowptr[src]; }
  inline __device__ __host__ vidType getDestination(vidType src, eidType edge) { return d_colidx[d_rowptr[src] + edge]; }
  inline __device__ __host__ vidType getAbsDestination(eidType abs_edge) { return d_colidx[abs_edge]; }
  inline __device__ __host__ vidType getEdgeDst(eidType edge) { return d_colidx[edge]; }
  inline __device__ __host__ eidType edge_begin(vidType src) { return d_rowptr[src]; }
  inline __device__ __host__ eidType edge_end(vidType src) { return d_rowptr[src+1]; }
  inline __device__ __host__ vlabel_t getData(vidType vid) { return d_labels[vid]; }
  void clean() {
    CUDA_SAFE_CALL(cudaFree(d_rowptr));
    CUDA_SAFE_CALL(cudaFree(d_colidx));
  }
  void clean_edgelist() {
    CUDA_SAFE_CALL(cudaFree(d_src_list));
    CUDA_SAFE_CALL(cudaFree(d_dst_list));
  }
  void init(Graph &g, int n, int m) {
    device_id = n;
    n_gpu = m;
    init(g);
  }
  void init(Graph &hg) {
    auto m = hg.num_vertices();
    auto nnz = hg.num_edges();
    num_vertices = m;
    num_edges = nnz;
    auto h_rowptr = hg.out_rowptr();
    auto h_colidx = hg.out_colidx();
    size_t mem_vert = size_t(m+1)*sizeof(eidType);
    size_t mem_edge = size_t(nnz)*sizeof(vidType);
    size_t mem_graph = mem_vert + mem_edge;
    size_t mem_el = mem_edge; // memory for the edgelist
    size_t mem_all = mem_graph + mem_el;
    auto mem_gpu = get_gpu_mem_size();
    Timer t;
    if (mem_all > mem_gpu) {
      std::cout << "Allocating graph memory using CUDA unified memory\n";
      CUDA_SAFE_CALL(cudaMallocManaged(&d_colidx, nnz * sizeof(vidType)));
      std::copy(h_colidx, h_colidx+nnz, d_colidx);
      if (mem_vert + mem_el < mem_gpu) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr, (m + 1) * sizeof(eidType)));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        t.Start();
        CUDA_SAFE_CALL(cudaMemcpy(d_rowptr, h_rowptr, (m + 1) * sizeof(eidType), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        t.Stop();
        std::cout << "Time on copying vertex pointers to GPU: " << t.Seconds() << " sec\n";
      } else {
        CUDA_SAFE_CALL(cudaMallocManaged(&d_rowptr, (m + 1) * sizeof(eidType)));
        std::copy(h_rowptr, h_rowptr+m+1, d_rowptr);
        //CUDA_SAFE_CALL(cudaMemPrefetchAsync(d_colidx, nnz*sizeof(vidType), 0, NULL));
        //CUDA_SAFE_CALL(cudaMemPrefetchAsync(d_rowptr, (m+1)*sizeof(eidType), 0, NULL));
      }
    } else {
      std::cout << "Allocating graph memory on GPU" << device_id << "\n";
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr, (m + 1) * sizeof(eidType)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx, nnz * sizeof(vidType)));
      if (hg.has_vlabel()) CUDA_SAFE_CALL(cudaMalloc((void **)&d_labels, m * sizeof(vlabel_t)));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      t.Start();
      CUDA_SAFE_CALL(cudaMemcpy(d_rowptr, h_rowptr, (m + 1) * sizeof(eidType), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_colidx, h_colidx, nnz * sizeof(vidType), cudaMemcpyHostToDevice));
      if (hg.has_vlabel()) CUDA_SAFE_CALL(cudaMemcpy(d_labels, hg.getVlabelPtr(), m * sizeof(vlabel_t), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      t.Stop();
      std::cout << "Time on copying graph to GPU" << device_id << ": " << t.Seconds() << " sec\n";
    }
  }
  // this is for single-GPU only
  size_t init_edgelist(Graph &hg, bool sym_break = false, bool ascend = false) {
    auto nnz = num_edges;
    if (sym_break) nnz = nnz/2;
    size_t mem_el = size_t(nnz)*sizeof(vidType);
    auto mem_gpu = get_gpu_mem_size();
    //size_t mem_graph_el = size_t(num_vertices+1)*sizeof(eidType) + size_t(2)*size_t(nnz)*sizeof(vidType);
    if (mem_el > mem_gpu) {
      std::cout << "Allocating edgelist (size = " << nnz << ") using CUDA unified memory\n";
      CUDA_SAFE_CALL(cudaMallocManaged(&d_src_list, nnz * sizeof(vidType)));
      if (!sym_break) d_dst_list = d_colidx;
      else CUDA_SAFE_CALL(cudaMallocManaged(&d_dst_list, nnz * sizeof(vidType)));
      init_edgelist_um(hg, sym_break);
      //CUDA_SAFE_CALL(cudaDeviceSynchronize());
      //Timer t;
      //t.Start();
      //CUDA_SAFE_CALL(cudaMemPrefetchAsync(d_src_list, nnz*sizeof(vidType), 0, NULL));
      //CUDA_SAFE_CALL(cudaMemPrefetchAsync(d_dst_list, nnz*sizeof(vidType), 0, NULL));
      //CUDA_SAFE_CALL(cudaDeviceSynchronize());
      //t.Stop();
    } else {
      hg.init_edgelist(sym_break, ascend);
      copy_edgelist_to_device(nnz, hg.get_src_ptr(), hg.get_dst_ptr(), sym_break);
    }
    return nnz;
  }
  void copy_edgelist_to_device(size_t nnz, Graph &hg, bool sym_break = false) {
    copy_edgelist_to_device(nnz, hg.get_src_ptr(), hg.get_dst_ptr(), sym_break);
  }
  void copy_edgelist_to_device(size_t nnz, vidType* h_src_list, vidType* h_dst_list, bool sym_break) {
    eidType n_tasks_per_gpu = eidType(nnz-1) / eidType(n_gpu) + 1;
    if (!sym_break) d_dst_list = d_colidx + device_id * n_tasks_per_gpu;
    eidType start = device_id * n_tasks_per_gpu;
    eidType num = n_tasks_per_gpu;
    if (start + num > nnz) num = nnz - start;
    std::cout << "Allocating edgelist on GPU" << device_id << " size = " << num << "\n";
    Timer t;
    t.Start();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_src_list, num * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_src_list, h_src_list+start, num * sizeof(vidType), cudaMemcpyHostToDevice));
    if (sym_break) {
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_dst_list, num * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMemcpy(d_dst_list, h_dst_list+start, num * sizeof(vidType), cudaMemcpyHostToDevice));
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t.Stop();
    std::cout << "Time on copying edgelist to GPU" << device_id << ": " << t.Seconds() << " sec\n";
  }
  void copy_edgelist_to_device(std::vector<eidType> lens, std::vector<vidType*> &srcs, std::vector<vidType*> &dsts) {
    Timer t;
    t.Start();
    vidType* src_ptr = srcs[device_id];
    vidType* dst_ptr = dsts[device_id];
    auto num = lens[device_id];
    //std::cout << "src_ptr = " << src_ptr << " dst_ptr = " << dst_ptr << "\n";
    std::cout << "Allocating edgelist on GPU" << device_id << " size = " << num << "\n";
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_src_list, num * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_src_list, src_ptr, num * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_dst_list, num * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_dst_list, dst_ptr, num * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t.Stop();
    std::cout << "Time on copying edgelist to GPU" << device_id << ": " << t.Seconds() << " sec\n";
  }
  void init_edgelist_um(Graph &g, bool sym_break = false) {
    Timer t;
    t.Start();
    size_t i = 0;
    for (vidType v = 0; v < g.V(); v ++) {
      for (auto u : g.N(v)) {
        assert(u != v);
        if (sym_break && v < u) break;  
        d_src_list[i] = v;
        if (sym_break) d_dst_list[i] = u;
        i ++;
      }
    }
    t.Stop();
    std::cout << "Time generating the edgelist on CUDA unified memory: " << t.Seconds() << " sec\n";
  }

  // using a warp to compute the intersection of the neighbor lists of two vertices
  inline __device__ vidType warp_intersect(vidType src, vidType dst) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
    vidType count = 0;
    assert(src != dst);
    vidType src_size = getOutDegree(src);
    vidType dst_size = getOutDegree(dst);
    if (src_size == 0 || dst_size == 0) return 0;
    vidType* lookup = d_colidx + edge_begin(src);
    vidType* search = d_colidx + edge_begin(dst);
    vidType lookup_size = src_size;
    vidType search_size = dst_size;
    if (src_size > dst_size) {
      auto temp = lookup;
      lookup = search;
      search = temp;
      search_size = src_size;
      lookup_size = dst_size;
    }
    for (vidType i = thread_lane; i < lookup_size; i += WARP_SIZE) {
      auto key = lookup[i];
      if (binary_search(search, key, search_size))
        count += 1;
    }
    return count;
  }

  // using a CTA to compute the intersection of the neighbor lists of two vertices
  inline __device__ vidType cta_intersect(vidType src, vidType dst) {
    vidType count = 0;
    assert(src != dst);
    vidType src_size = getOutDegree(src);
    vidType dst_size = getOutDegree(dst);
    if (src_size == 0 || dst_size == 0) return 0;
    vidType* lookup = d_colidx + edge_begin(src);
    vidType* search = d_colidx + edge_begin(dst);
    vidType lookup_size = src_size;
    vidType search_size = dst_size;
    if (src_size > dst_size) {
      auto temp = lookup;
      lookup = search;
      search = temp;
      search_size = src_size;
      lookup_size = dst_size;
    }
    for (vidType i = threadIdx.x; i < lookup_size; i += BLOCK_SIZE) {
      auto key = lookup[i];
      if (binary_search(search, key, search_size))
        count += 1;
    }
    return count;
  }

  // using a warp to compute the intersection of the neighbor lists of two vertices with caching
  inline __device__ vidType warp_intersect_cache(vidType src, vidType dst) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
    int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
    __shared__ vidType cache[BLOCK_SIZE];
    vidType count = 0;
    assert(src != dst);
    vidType src_size = getOutDegree(src);
    vidType dst_size = getOutDegree(dst);
    if (src_size == 0 || dst_size == 0) return 0;
    vidType* lookup = d_colidx + edge_begin(src);
    vidType* search = d_colidx + edge_begin(dst);
    vidType lookup_size = src_size;
    vidType search_size = dst_size;
    if (src_size > dst_size) {
      auto temp = lookup;
      lookup = search;
      search = temp;
      search_size = src_size;
      lookup_size = dst_size;
    }
    cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
    __syncwarp();
    for (vidType i = thread_lane; i < lookup_size; i += WARP_SIZE) {
      auto key = lookup[i];
      if (binary_search_2phase(search, cache, key, search_size))
        count += 1;
    }
    return count;
  }

  // using a cta compute the intersection of the neighbor lists of two vertices with caching
  inline __device__ vidType cta_intersect_cache(vidType src, vidType dst) {
    __shared__ vidType cache[BLOCK_SIZE];
    vidType count = 0;
    assert(src != dst);
    vidType src_size = getOutDegree(src);
    vidType dst_size = getOutDegree(dst);
    if (src_size == 0 || dst_size == 0) return 0;
    vidType* lookup = d_colidx + edge_begin(src);
    vidType* search = d_colidx + edge_begin(dst);
    vidType lookup_size = src_size;
    vidType search_size = dst_size;
    if (src_size > dst_size) {
      auto temp = lookup;
      lookup = search;
      search = temp;
      search_size = src_size;
      lookup_size = dst_size;
    }
    cache[threadIdx.x] = search[threadIdx.x * search_size / BLOCK_SIZE];
    __syncthreads();
    for (vidType i = threadIdx.x; i < lookup_size; i += BLOCK_SIZE) {
      auto key = lookup[i];
      if (binary_search_2phase_cta(search, cache, key, search_size))
        count += 1;
    }
    return count;
  }
};

