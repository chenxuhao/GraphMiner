// warp-centric vertex-parallel: each warp takes one vertex
__global__ void __launch_bounds__(BLK_SZ, 8)
rectangle_warp_vertex_nested(vidType nv, GraphGPU g, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id   = thread_id / WARP_SIZE;
  int num_warps = (BLK_SZ / WARP_SIZE) * gridDim.x;
  AccType counter = 0; 
  for (vidType v0 = warp_id; v0 < nv; v0 += num_warps) {
    auto v0_size = g.get_degree(v0);
    auto v0_ptr = g.N(v0);
    for (vidType idx0 = 0; idx0 < v0_size; idx0++) {
      auto v1 = v0_ptr[idx0];
      if (v1 >= v0) break;
      auto v1_size = g.get_degree(v1);
      for (vidType idx1 = 0; idx1 < v0_size; idx1++) {
        auto v2 = v0_ptr[idx1];
        if (v2 >= v1) break;
        auto v2_size = g.get_degree(v2);
        counter += intersect_num(g.N(v1), v1_size, g.N(v2), v2_size, v0);
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

