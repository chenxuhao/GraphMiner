// Copyright 2020 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
// edge parallel: each warp takes one edge
__global__ void rectangle_warp_edge_iterative(eidType ne, int k, GraphGPU g, vidType *vlist, unsigned max_deg, AccType *total) {
  unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  unsigned num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  unsigned depth = 1;
  Status state = Idle;
  vidType idx[MAX_PATTERN_SIZE-2];
  vidType stack[MAX_PATTERN_SIZE];
  __shared__ vidType vlist_sizes[WARPS_PER_BLOCK][MAX_PATTERN_SIZE-2];
 
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    assert(v1 < v0);
    depth = 1;
    stack[0] = v0;
    stack[1] = v1;
    state = Extending;
    auto begin = g.edge_begin(v0);
    while(1) {
      if (depth == k-2) { // found a match
        assert(depth == 2);
        auto u = stack[2];
        vidType u_size = g.getOutDegree(u);
        vidType v1_size = g.getOutDegree(v1);
        auto count = intersect_num(g.N(u), u_size, g.N(v1), v1_size, v0);
        auto sum = warp_reduce<unsigned>(count);
        if (thread_lane == 0) atomicAdd(total, sum);
        depth --; // backtrack
        state = IteratingEdge;
      } else if (state == Extending) {
        vidType v_size = g.getOutDegree(v0);
        unsigned count = count_smaller(v1, g.N(v0), v_size);
        if (thread_lane == 0)
          vlist_sizes[warp_lane][depth-1] = count;
        __syncwarp();
        idx[depth-1] = 0;
      }
      if (depth == 0) break; 
      if (idx[depth-1] == vlist_sizes[warp_lane][depth-1]) {
        if (depth == 1) break; // this subtree is done
        else { // backtrack
          depth --;
          state = IteratingEdge;
        }
      } else {
        auto i = idx[depth-1];
        auto w = g.getEdgeDst(begin+i);
        idx[depth-1] = i + 1;
        depth ++;
        stack[depth] = w;
        state = Extending;
      }
    } // end while
  }
}
