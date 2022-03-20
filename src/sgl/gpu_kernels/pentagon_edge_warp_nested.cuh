// edge parallel: each warp takes one edge
__global__ void __launch_bounds__(BLK_SZ, 6)
pentagon_warp_edge_nested(eidType ne, GraphGPU g, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  AccType counter = 0; 
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto begin = g.edge_begin(v0);
    for (vidType idx2 = 0; idx2 < v0_size; idx2++) {
      vidType v2 = g.N(v0)[idx2];
      if (v2 >= v1) break;
      vidType v2_size = g.getOutDegree(v2);
      for (vidType idx3 = 0; idx3 < v2_size; idx3++) {
        vidType v3 = g.N(v2)[idx3];
        if (v3 >= v0) break;
        if (v3 == v1) continue;
        vidType v3_size = g.getOutDegree(v3);
        // v4∈ adj(v1)∩ adj(v3) && v4 < v0 && v4 != v2
        counter += intersect_num(g.N(v1), v1_size, g.N(v3), v3_size, v0, v2);
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}
