// steal tasks from another warp in the same thread block who has the most task remaining
__forceinline__ __device__ unsigned work_stealing(Status *status, vidType *vlist_sizes, vidType *idx, vidType *v0, vidType *v1) {
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  //vidType max_size = 0;
  //unsigned victim = WARPS_PER_BLOCK;
  __shared__ bool succeed[WARPS_PER_BLOCK];     // whether the stealing is successful
  __shared__ vidType max_size[WARPS_PER_BLOCK]; // the remaining number of tasks of the victim warp
  //__shared__ vidType count[WARPS_PER_BLOCK];    // the total number of tasks of the victim warp
  __shared__ unsigned victim[WARPS_PER_BLOCK];  // warp_lane of the victim warp
  __shared__ unsigned stealer;

  // find a victim who has the most tasks
  if (thread_lane == 0) {
    succeed[warp_lane] = false;
    vlist_sizes[warp_lane] = 0;
    max_size[warp_lane] = 0;
    victim[warp_lane] = WARPS_PER_BLOCK;
    for (int wid = 0; wid < WARPS_PER_BLOCK; wid++) {
      if (wid != warp_lane && status[wid] == Working) {
        vlist_sizes[warp_lane] = vlist_sizes[wid];
        auto num = vlist_sizes[wid] - idx[wid];
        if (num > max_size[warp_lane]) {
          max_size[warp_lane] = num;
          victim[warp_lane] = wid;
        }
      }
    }
  }
  __syncwarp();
  vidType current_v0, current_v1;
  current_v0 = v0[victim[warp_lane]];
  current_v1 = v1[victim[warp_lane]];

  // only one warp in the thread block can steal at a time
  if (stealer >= WARPS_PER_BLOCK || status[stealer] != Idle)
    if (thread_lane == 0) stealer = warp_lane;

  // update stealer's status
  auto steal_num = max_size[warp_lane]/2;
  if (warp_lane == stealer && steal_num > 5 && victim[warp_lane] != WARPS_PER_BLOCK) {
    //if (thread_lane == 0) printf("%d tasks to steal from warp %d to warp %d\n", steal_num, victim[warp_lane], warp_lane);
    // split half of the remaining tasks from the victim
    if (vlist_sizes[victim[warp_lane]] == vlist_sizes[warp_lane]) {
      auto old_value = atomicCAS(&vlist_sizes[victim[warp_lane]], vlist_sizes[warp_lane], vlist_sizes[warp_lane] - steal_num);
      if (old_value == vlist_sizes[warp_lane]) {
      //if (atomicCAS(&stealer, warp_lane, WARPS_PER_BLOCK) == warp_lane) {
        // make sure the vistim is still working on the current task
        if (current_v0 == v0[victim[warp_lane]] && current_v1 == v1[victim[warp_lane]]) {
          succeed[warp_lane] = true;
          //printf("%d tasks stolen from warp %d to warp %d\n", steal_num, victim[warp_lane], warp_lane);
          //vlist_sizes[victim[warp_lane]] = vlist_sizes[warp_lane] - steal_num;
          status[warp_lane] = ReWorking;
          //vlist_sizes[warp_lane] = count[warp_lane];
          idx[warp_lane] = vlist_sizes[warp_lane] - steal_num;
          v0[warp_lane] = current_v0;
          v1[warp_lane] = current_v1;
        } else { // failed, roll back
          //printf("stealing from warp %d to warp %d failed\n", victim[warp_lane], warp_lane);
          //if (thread_lane == 0) vlist_sizes[victim[warp_lane]] = old_value;
        }
      } else { // failed, roll back
        //printf("stealing from warp %d to warp %d failed due to non-stealer\n", victim[warp_lane], warp_lane);
        if (thread_lane == 0) vlist_sizes[victim[warp_lane]] = old_value;
      }
    }
  }
  __syncwarp();
  if (!succeed[warp_lane] && thread_lane == 0) {
    victim[warp_lane] = WARPS_PER_BLOCK;
    vlist_sizes[warp_lane] = 0;
  }
  __syncwarp();
  return victim[warp_lane];
}

// edge parallel: each warp takes one edge
__global__ void rectangle_warp_edge_nested_balanced(eidType ne, GraphGPU g, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  unsigned num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps

  __shared__ bool all_done; // all warp in the same thread block are done
  __shared__ int depth[WARPS_PER_BLOCK];
  __shared__ Status status[WARPS_PER_BLOCK];
  __shared__ vidType v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ vidType v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  __shared__ eidType eid[WARPS_PER_BLOCK];
  __shared__ vidType vlist_sizes[WARPS_PER_BLOCK];
  __shared__ vidType idx[WARPS_PER_BLOCK];
  if (thread_lane == 0) {
    eid[warp_lane] = warp_id;
    vlist_sizes[warp_lane] = 0;
    idx[warp_lane] = 0;
    if (warp_id >= ne) status[warp_lane] = Idle;
    else status[warp_lane] = Working;
  }
  if (threadIdx.x == 0) all_done = false;
  //if (threadIdx.x < WARPS_PER_BLOCK) {
  //  v0[threadIdx.x] = g.get_src(eid[thread_id]);
  //  v1[threadIdx.x] = g.get_src(eid[thread_id]);
  //}
  __syncthreads();

  AccType counter = 0;
  while (!all_done) {
    // stage 1: get a task (i.e. an edge in the edgelist)
    unsigned count = 0;
    if (status[warp_lane] == Working) {
      if (thread_lane == 0) {
        v0[warp_lane] = g.get_src(eid[warp_lane]);
        v1[warp_lane] = g.get_dst(eid[warp_lane]);
        v0_size[warp_lane] = g.getOutDegree(v0[warp_lane]);
        v1_size[warp_lane] = g.getOutDegree(v1[warp_lane]);
      }
      __syncwarp();
      count = count_smaller(v1[warp_lane], g.N(v0[warp_lane]), v0_size[warp_lane]);
      if (thread_lane == 0) {
        //status[warp_lane] = Working;
        eid[warp_lane] += num_warps;
        //vlist_sizes[warp_lane] = count;
        if (atomicCAS(&vlist_sizes[warp_lane], 0, count) != 0);
          //printf("warn: 0\n");
        idx[warp_lane] = 0;
        //printf("warp %d v0 %d (deg=%d) v1 %d (deg=%d) num_tasks %d\n", warp_id, v0[warp_lane], v0_size[warp_lane], v1[warp_lane], v1_size[warp_lane], count);
      }
      //unsigned active = __activemask();
      __syncwarp();
    }

    // stage 2: search the sub-tree of the current task
    if (status[warp_lane] != Idle) {
      while (idx[warp_lane] < vlist_sizes[warp_lane]) {
        vidType v2 = g.N(v0[warp_lane])[idx[warp_lane]];
        vidType v2_size = g.getOutDegree(v2);
        counter += intersect_num(g.N(v1[warp_lane]), v1_size[warp_lane], g.N(v2), v2_size, v0[warp_lane]);
        if (thread_lane == 0) idx[warp_lane]++;
        __syncwarp();
      }
      // self work done
      if (eid[warp_lane] >= ne) {
        if (thread_lane == 0) {
          status[warp_lane] = Idle;
          //vlist_sizes[warp_lane] = 0;
          auto old_val = atomicCAS(&vlist_sizes[warp_lane], count, 0);
          if (status[warp_lane] == Working && old_val != count)
            printf("warn: old_val = %d count = %d\n", old_val, count);
          idx[warp_lane] = 0;
        }
      }
    }
    __syncwarp();

    // stage 3: check for the termination condition
    if (status[warp_lane] == Idle) {
      // check if all warps are done
      bool others_done = true;
      for (int i = 0; i < WARPS_PER_BLOCK; i++) {
        if (status[i] != Idle) {
          others_done = false;
          break;
        }
      }
      if (others_done) {
        if (thread_lane == 0) all_done = true;
      ///*
      } else { // if another warp is still running, steal some work form its task queue
        auto victim = work_stealing(status, vlist_sizes, idx, v0, v1);
        assert(victim != warp_lane);
        if (victim != WARPS_PER_BLOCK) { // stealing succeed
          if (thread_lane == 0) {
            v0_size[warp_lane] = g.getOutDegree(v0[warp_lane]);
            v1_size[warp_lane] = g.getOutDegree(v1[warp_lane]);
            //printf("warp %d steal from warp %d: task v0 %d v1 %d\n", warp_id, warp_id - warp_lane + victim, v0[warp_lane], v1[warp_lane]);
          }
          __syncwarp();
        }
        //*/
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}
