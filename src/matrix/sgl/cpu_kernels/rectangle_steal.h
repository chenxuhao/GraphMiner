bool computation_end = false; // shared by all threads
vidType num_top_level_tasks = g.V();
vidType tasks_per_thread = (num_top_level_tasks-1)/num_threads + 1;
std::vector<Thread_private_data> thread_private_data(num_threads);
//std::vector<int> depths(num_threads, 0);
//std::vector<std::vector<vidType>> sizes(num_threads);
//std::vector<std::vector<vidType>> idx(num_threads);

#pragma omp parallel num_threads(num_threads) reduction(+:counter)
{
  int tid = omp_get_thread_num();
  auto& tpd = thread_private_data[tid];
  tpd.sizes.resize(3);
  tpd.idx.resize(3);
  vidType start_index = tid * tasks_per_thread;
  if (start_index < num_top_level_tasks) {
    tpd.sizes[0] = tasks_per_thread;
    if (start_index+tasks_per_thread > num_top_level_tasks)
      tpd.sizes[0] = num_top_level_tasks - start_index;
  }
  // set thread to start working
  if (tpd.sizes[0] > 0)
    lb.activate_thread(tid);
#pragma omp barrier
  tpd.idx[0] = 0;
  printf("thread %d start from %d, num_tasks %d\n", tid, start_index, tpd.sizes[0]);
  while (!computation_end) {
    if (!lb.thread_working(tid)) { // check if all threads are done
      if (num_threads > 1 && !computation_end)
        lb.task_schedule(tid, thread_private_data);
      if (tid == 0) {
        if (lb.all_threads_idle()) computation_end = true;
      }
    } else { // process the next task
      tpd.depth = 0;
      if (num_threads > 1) lb.task_schedule(tid, thread_private_data);
      vidType v0 = start_index + tpd.idx[0];
      //printf("thread %d v0 %d\n", tid, v0);
      tpd.idx[0] ++;
      auto y0 = g.N(v0);
      auto y0f0 = bounded(y0, v0);
      tpd.sizes[1] = y0f0.size();
      tpd.idx[1] = 0;
      while (tpd.idx[1] < tpd.sizes[1]) {
        tpd.depth = 1;
        vidType v1 = y0f0.begin()[tpd.idx[1]];
        tpd.idx[1]++;
        //printf("\t thread %d v0 %d v1 %d\n", tid, v0, v1);
        auto y1 = g.N(v1);
        auto y0f1 = bounded(y0, v1);
        tpd.sizes[2] = y0f1.size();
        tpd.idx[2] = 0;
        while (tpd.idx[2] < tpd.sizes[2]) {
          tpd.depth = 2;
          vidType v2 = y0f1.begin()[tpd.idx[2]];
          tpd.idx[2]++;
          //printf("\t\t thread %d v0 %d v1 %d v2 %d\n", tid, v0, v1, v2);
          counter += intersection_num(y1, g.N(v2), v0);
        }
        tpd.idx[2] = 0;
        tpd.sizes[2] = 0;
      }
      tpd.idx[1] = 0;
      tpd.sizes[1] = 0;
      tpd.depth = 0;
      if (tpd.idx[0] >= tpd.sizes[0]) {
        lb.deactivate_thread(tid);
        tpd.idx[0] = 0;
        tpd.sizes[0] = 0;
        //printf("thread %d completed\n", tid);
      }
    }
  } // end while
} // end omp parallel
