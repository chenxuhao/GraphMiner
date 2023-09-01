// Copyright 2020 MIT
// Contact: Xuhao Chen <cxh@mit.edu>

__global__ void rectangle_warp_edge_nested_cmap(eidType ne, vidType n, GraphGPU g, cmap_gpu cmap, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  unsigned num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  __shared__ vidType bin_counts[WARPS_PER_BLOCK*NUM_BUCKETS];
  cmap.init_bin_counts(bin_counts);
 
  AccType counter = 0; 
  // edge parallel: each warp takes one edge
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid); // v0
    auto v1 = g.get_dst(eid); // v1
    assert(v1 < v0); // symmetry breaking
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    vidType *v0_ptr = g.N(v0);
    vidType *v1_ptr = g.N(v1);
    for (vidType i = thread_lane; i < v1_size; i+=WARP_SIZE) {
      vidType u = v1_ptr[i];
      int is_smaller = u < v0 ? 1 : 0;
      if (is_smaller) cmap.insert(warp_id, u, bin_counts);
      unsigned active = __activemask();
      unsigned mask = __ballot_sync(active, is_smaller);
      if (mask != FULL_MASK) break;
    }
    __syncwarp();
    for (vidType idx1 = 0; idx1 < v0_size; idx1++) {
      vidType v2 = v0_ptr[idx1]; // v2
      if (v2 >= v1) break; // symmetry breaking
      vidType v2_size = g.getOutDegree(v2);
      vidType *v2_ptr = g.N(v2);
      for (vidType idx2 = thread_lane; idx2 < v2_size; idx2+=WARP_SIZE) {
        vidType v3 = v2_ptr[idx2]; // v3
        int is_smaller = v3 < v0 ? 1 : 0;
        if (is_smaller && cmap.lookup(warp_id, v3, bin_counts)) counter++;
        unsigned active = __activemask();
        unsigned mask = __ballot_sync(active, is_smaller);
        if (mask != FULL_MASK) break;
      }
    }
    __syncwarp();
    cmap.init_bin_counts(bin_counts);
    /*
    for (vidType i = thread_lane; i < v1_size; i+=WARP_SIZE) {
      vidType u = v1_ptr[i];
      int is_smaller = u < v0 ? 1 : 0;
      if (is_smaller) cmap.clean(u, bin_counts);
      unsigned active = __activemask();
      unsigned mask = __ballot_sync(active, is_smaller);
      if (mask != FULL_MASK) break;
    }
    __syncwarp();
    */
  }
  // count reduction
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

__global__ void rectangle_warp_vertex_nested_cmap(vidType nv, vidType n, GraphGPU g, cmap_vt *cmaps, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  unsigned num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  cmap_vt *cmap = &cmaps[n*warp_id];
  AccType counter = 0; 
  // edge parallel: each warp takes one edge
  for (vidType v0 = warp_id; v0 < nv; v0 += num_warps) {
    vidType v0_size = g.getOutDegree(v0);
    vidType *v0_ptr = g.N(v0);
    for (vidType idx0 = 0; idx0 < v0_size; idx0++) {
      auto v1 = v0_ptr[idx0]; // v1
      if (v1 >= v0) break; // symmetry breaking
      vidType v1_size = g.getOutDegree(v1);
      for (vidType i = thread_lane; i < v1_size; i+=WARP_SIZE) {
        vidType u = g.N(v1)[i];
        if (u >= v0) break;
        cmap[u]=1;
      }
      for (vidType idx1 = 0; idx1 < v0_size; idx1++) {
        vidType v2 = v0_ptr[idx1]; // v2
        if (v2 >= v1) break; // symmetry breaking
        vidType v2_size = g.getOutDegree(v2);
        for (vidType idx2 = thread_lane; idx2 < v2_size; idx2+=WARP_SIZE) {
          vidType v3 = g.N(v2)[idx2]; // v3
          if (v3 >= v0) break; // symmetry breaking
          if (cmap[v3]==1) counter++;
        }
      }
      for (vidType i = thread_lane; i < v1_size; i+=WARP_SIZE) {
        vidType u = g.N(v1)[i];
        if (u >= v0) break;
        cmap[u]=0;
      }
    }
  }
  // count reduction
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

