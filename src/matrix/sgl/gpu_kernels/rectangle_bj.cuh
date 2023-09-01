/*
__device__ sort_by_key(int n, int *d_keys, int *d_values) {
  assert(n < WARP_SIZE*ITEMS_PER_THREAD);
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int nwarps_blk  = BLK_SZ / WARP_SIZE;
  int nwarps_all  = nwarps_blk * gridDim.x;
 
  __shared__ int keys[ITEMS_PER_THREAD];
  __shared__ int values[ITEMS_PER_THREAD];
  // load from global into shared variable
  keys[threadIdx.x] = ddata[threadIdx.x];
  unsigned int bitmask = 1<<LOWER_BIT;
  unsigned int offset  = 0;
  unsigned int thrmask = 0xFFFFFFFFU << threadIdx.x;
  unsigned int mypos;
  //  for each LSB to MSB
  for (int i = LOWER_BIT; i <= UPPER_BIT; i++){
    unsigned int mydata = sdata[((WSIZE-1)-threadIdx.x)+offset];
    unsigned int mybit  = mydata&bitmask;
    // get population of ones and zeroes (cc 2.0 ballot)
    unsigned int ones = __ballot(mybit); // cc 2.0
    unsigned int zeroes = ~ones;
    offset ^= WSIZE; // switch ping-pong buffers
    // do zeroes, then ones
    if (!mybit) // threads with a zero bit
      // get my position in ping-pong buffer
      mypos = __popc(zeroes&thrmask);
    else        // threads with a one bit
      // get my position in ping-pong buffer
      mypos = __popc(zeroes)+__popc(ones&thrmask);
    // move to buffer  (or use shfl for cc 3.0)
    sdata[mypos-1+offset] = mydata;
    // repeat for next bit
    bitmask <<= 1;
  }
  // save results to global
  ddata[threadIdx.x] = sdata[threadIdx.x+offset];
}

__device__ sort_by_key(int n, int *d_keys, int *d_values) {
  int block_begin = blockIdx.x * (BLK_SZ * ITEMS_PER_THREAD);

  // Allocate shared memory
  __shared__ union {
    typename BlockRadixSort::TempStorage  sort;
    typename BlockLoad::TempStorage       load;
    typename BlockStore::TempStorage      store;
  } temp_storage;

  BlockLoad(temp_storage.load).Load(d_keys_result + block_offset, thread_keys);
  BlockLoad(temp_storage.load).Load(d_values_result + block_offset, thread_values);
  __syncthreads();
  BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(d_keys_result, thread_values);
  __syncthreads(); 
  // --- Store the sorted segment
  BlockStoreIntT(temp_storage.storeInt).Store(d_keys_result + block_offset, thread_keys);
  BlockStoreFloatT(temp_storage.storeFloat).Store(d_values_result + block_offset, thread_values);
}
*/
/*
// warp-centric vertex-parallel: each warp takes one vertex
__global__ void __launch_bounds__(BLK_SZ, 8)
rectangle_warp_vertex_nested(vidType nv, GraphGPU g, AccType *total, vidType *wa, vidType *wb, vidType *wc) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int nwarps_blk  = BLK_SZ / WARP_SIZE;
  int nwarps_all  = nwarps_blk * gridDim.x;
  vidType begin = warp_id*max_deg*max_deg;
  __shared__ int len[BLK_SZ/WARP_SIZE];
  if (thread_lane == 0) len[warp_lane] = 0;
  __syncwarp();

  // listing wedges incident to v0
  for (vidType v0 = warp_id; v0 < nv; v0 += nwarps_all) {
    vidType v0_size = g.getOutDegree(v0);
    auto v0_ptr = g.N(v0);
    for (vidType idx0 = 0; idx0 < v0_size; idx0++) {
      auto v1 = v0_ptr[idx0];
      if (v1 >= v0) break;
      vidType v1_size = g.getOutDegree(v1);
      auto count = count_smaller(v0, g.N(v1), v1_size, wb[begin+len[warp_lane]]);
      if (thread_lane == 0) len[warp_lane] += count;
      __syncwarp();
    }

    // group wedges by v2, and then sort v1 in each group
    for (int wid = 0; wid < nwarps_blk; wid++) {
      sort_by_key(len[wid], wb, wa);
      __syncthreads();
    }
  }
}
*/
__global__ void __launch_bounds__(BLK_SZ, 8)
wedge_counting(vidType v, vidType nv, GraphGPU g, int *num_wedges) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;
  __shared__ int len[BLK_SZ/WARP_SIZE];
  if (thread_lane == 0) len[warp_lane] = 0;
  __syncwarp();

  // count the number of wedges incident to v0
  //for (vidType v0 = warp_id; v0 < nv; v0 += nwarps_all) {
    vidType v0 = v + warp_id;
    if (v0 > nv) return;
    vidType v0_size = g.getOutDegree(v0);
    auto v0_ptr = g.N(v0);
    for (vidType idx0 = 0; idx0 < v0_size; idx0++) {
      auto v1 = v0_ptr[idx0];
      if (v1 >= v0) break;
      vidType v1_size = g.getOutDegree(v1);
      int count = count_smaller(v0, g.N(v1), v1_size);
      if (thread_lane == 0) len[warp_lane] += count;
      unsigned active = __activemask();
      __syncwarp(active);
    }
    if (thread_lane == 0) num_wedges[warp_id] = len[warp_lane];
  //}
}
/*
__global__ void //__launch_bounds__(BLK_SZ, 8)
wedge_listing(vidType v, vidType nv, GraphGPU g, vidType max_deg, vidType *w1, vidType *w2, int *num_wedges) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;
  //int nwarps_blk  = BLK_SZ / WARP_SIZE;
  //int nwarps_all  = nwarps_blk * gridDim.x;
  const vidType begin = warp_id*max_deg*max_deg;
  __shared__ int len[BLK_SZ/WARP_SIZE];
  if (thread_lane == 0) len[warp_lane] = 0;
  __syncwarp();

  // listing wedges incident to v0
  //for (vidType v0 = warp_id; v0 < nv; v0 += nwarps_all) {
    vidType v0 = v + warp_id;
    if (v0 > nv) return;
    vidType v0_size = g.getOutDegree(v0);
    auto v0_ptr = g.N(v0);
    for (vidType idx0 = 0; idx0 < v0_size; idx0++) {
      auto v1 = v0_ptr[idx0];
      if (v1 >= v0) break;
      vidType v1_size = g.getOutDegree(v1);
      auto offset = begin + len[warp_lane];
      int count = list_smaller(v0, g.N(v1), v1_size, w2+offset);
      assert (count >= 0 && count <= max_deg);
      for (auto i = thread_lane; i < count; i += WARP_SIZE)
        w1[offset+i] = v1;
      if (thread_lane == 0) len[warp_lane] += count;
      unsigned active = __activemask();
      __syncwarp(active);
    }
    if (thread_lane == 0) num_wedges[warp_id] = len[warp_lane];
  //}
}
*/
__global__ void __launch_bounds__(BLK_SZ, 8)
wedge_listing(vidType v, vidType nv, GraphGPU g, int *indices, vidType *w1, vidType *w2) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;
  //int nwarps_blk  = BLK_SZ / WARP_SIZE;
  //int nwarps_all  = nwarps_blk * gridDim.x;
  vidType begin = indices[warp_id];
  __shared__ int len[BLK_SZ/WARP_SIZE];
  if (thread_lane == 0) len[warp_lane] = 0;
  __syncwarp();

  // listing wedges incident to v0
  //for (vidType v0 = warp_id; v0 < nv; v0 += nwarps_all) {
    vidType v0 = v + warp_id;
    if (v0 > nv) return;
    vidType v0_size = g.getOutDegree(v0);
    auto v0_ptr = g.N(v0);
    for (vidType idx0 = 0; idx0 < v0_size; idx0++) {
      auto v1 = v0_ptr[idx0];
      if (v1 >= v0) break;
      vidType v1_size = g.getOutDegree(v1);
      auto offset = begin + len[warp_lane];
      int count = list_smaller(v0, g.N(v1), v1_size, w2+offset);
      for (auto i = thread_lane; i < count; i += WARP_SIZE)
        w1[offset+i] = v1;
      if (thread_lane == 0) len[warp_lane] += count;
      unsigned active = __activemask();
      __syncwarp(active);
    }
  //}
}

// enumerate rectangles
__global__ void //__launch_bounds__(BLK_SZ, 8)
//enumerate_rectangles(vidType max_deg, int *sizes, vidType *w1, vidType *w2, AccType *total) {
enumerate_rectangles(int *indices, int *sizes, vidType *w1, vidType *w2, AccType *total) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);

  __shared__ typename BlockReduce::TempStorage temp_storage;
  AccType counter = 0;
  auto len = sizes[warp_id];
  //vidType begin = warp_id*max_deg*max_deg;
  vidType begin = indices[warp_id];
  size_t offset = 0;

  while (offset < len) {
    auto v2 = w2[begin+offset]; // the group with the same v2 
    auto end = offset;
    while (end < len) {
      unsigned active = __activemask();
      __syncwarp(active);
      int flag = 0;
      auto pos = end + thread_lane;
      if (pos <len && w2[begin+pos] == v2) flag = 1;
      unsigned mask = __ballot_sync(active, flag);
      end += __popc(mask);
      if (mask != active) break;
    }
    for (; offset < end; offset += WARP_SIZE) {
      auto pos = offset + thread_lane;
      if (pos < end) {
        // list the matched subgraph here
        //counter += end - pos - 1;
        for (size_t i = pos+1; i < end; ++i) ++counter;
      }
    }
    offset = end;
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

