// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 8)
clique5_warp_edge_bitset(eidType ne, GraphGPU g, vidType *vmaps, MultiBitsets<> adj_lists, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  //int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  vidType *vmap   = &vmaps[int64_t(warp_id)*int64_t(max_deg)];
  size_t offset = warp_id * max_deg * ((max_deg-1)/32+1);
  AccType counter = 0;

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
      for (auto j = thread_lane; j < count; j += WARP_SIZE) {
        unsigned active = __activemask();
        bool flag = (j!=i) && binary_search(search, vmap[j], search_size);
        __syncwarp(active);
        adj_lists.warp_set(offset, i, j, flag);
      }
    }
    //auto count = construct_induced_graph(g, v0, v1, adj_lists, vlists);
    __syncwarp();
    auto nc = (count-1) / 32 + 1;
    for (vidType v2 = 0; v2 < count; v2++) {
      for (vidType v3 = 0; v3 < count; v3++) {
        if (adj_lists.get(offset, v2, v3)) {
          auto count1 = adj_lists.intersect_num(offset, nc, v2, v3);
          //if (thread_lane == 0) {
            //if (count1 > 0) printf("v0=%d, v1=%d, v2=%d, v3=%d, count=%d\n", v0, v1, vmap[v2], vmap[v3], count1);
            counter += count1;
          //}
        }
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

