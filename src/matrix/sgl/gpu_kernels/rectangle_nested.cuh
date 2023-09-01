// warp-centric edge-parallel: each warp takes one edge
__global__ void __launch_bounds__(BLK_SZ, 8)
rectangle_warp_edge_nested(eidType ne, GraphGPU g, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;
  AccType counter = 0; 
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    vidType* adj0 = g.N(v0);
    vidType* adj1 = g.N(v1);
    for (vidType i = 0; i < v0_size; i++) {
      vidType v2 = adj0[i];
      if (v2 >= v1) break;
      vidType v2_size = g.getOutDegree(v2);
      counter += intersect_num(adj1, v1_size, g.N(v2), v2_size, v0);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

