// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 8)
clique4_warp_edge_bitset(eidType ne, GraphGPU g, vidType *vmaps, MultiBitsets<> adj_lists, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;
  vidType *vmap   = &vmaps[int64_t(warp_id)*int64_t(max_deg)];
  AccType counter = 0;
  //__shared__ vidType cache[BLOCK_SIZE];

  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    vidType count = intersect(g.N(v0), v0_size, g.N(v1), v1_size, vmap);
    __syncwarp();
    for (vidType i = 0; i < count; i++) {
      auto search = g.N(vmap[i]);
      vidType search_size = g.getOutDegree(vmap[i]);
      //cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
      __syncwarp();
      //auto upper_bound = search[search_size-1];
      for (auto j = thread_lane; j < count; j += WARP_SIZE) {
        auto key = vmap[j];
        //bool found = (j!=i) && binary_search_2phase(search, cache, key, search_size);
        bool found = (j!=i) && binary_search(search, key, search_size);
        unsigned active = __activemask();
        __syncwarp(active);
        if (found) counter ++;
      }
    }
    //auto count = construct_induced_graph(g, v0, v1, adj_lists, vlists);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

