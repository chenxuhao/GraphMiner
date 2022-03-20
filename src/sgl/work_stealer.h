#pragma once
#include "common.h"

struct Thread_private_data {
  //bool is_working;
  //int thread_id;
  int task_split_level;// at which level the tasks are stolen
  int depth;           // current DFS traversal depth
  VertexList sizes;    // num_tasks in each level
  VertexList idx;      // current index in each level
};

class LoadBalancer {
  typedef enum { RT_WORK_REQUEST = 0, RT_WORK_RESPONSE = 1, GLOBAL_WORK_REQUEST = 2, GLOBAL_WORK_RESPONSE = 3} REQUEST_TYPE;
  typedef enum { MSG_DATA = 5, MSG_REQUEST = 6} MSG_TAG; //, TAG_TOKEN=12
  typedef enum { ARR = 12, RP = 13} DYN_SCHEME;
private:
  int num_threads;
  int scheme;
  int task_split_threshold;
  omp_lock_t lock;
  std::vector<bool> thread_is_working;
  std::vector<int> next_work_request;
  std::vector<std::deque<int> > message_queue;
  std::vector<std::vector<int> > requested_work_from;
  int all_idle_work_request_counter;
  //std::vector<omp_lock_t*> qlock;
  //std::map<int, int> global_work_response_counter;

public:
  LoadBalancer(int nt) : num_threads(nt), scheme(RP), task_split_threshold(2) {
    omp_init_lock(&lock);
    thread_is_working.resize(nt);
    next_work_request.clear();
    message_queue.clear();
    requested_work_from.clear();
    for (int i = 0; i < nt; i++) {
      thread_is_working[i] = false;
      int p =  (i + 1) % num_threads;
      next_work_request.push_back(p);
      std::deque<int> dq;
      message_queue.push_back(dq);
      std::vector<int> vc;
      requested_work_from.push_back(vc);
    }
    all_idle_work_request_counter = 0;
  }

  ~LoadBalancer(){ omp_destroy_lock(&lock); }

  void activate_thread(int tid) {
    omp_set_lock(&lock);
    thread_is_working[tid] = true;
    omp_unset_lock(&lock);
  }
  void deactivate_thread(int tid) {
    omp_set_lock(&lock);
    thread_is_working[tid] = false;
    omp_unset_lock(&lock);
  }

  void thread_start_working(int tid) {
    omp_set_lock(&lock);
    thread_is_working[tid] = true;
    omp_unset_lock(&lock);
  }

  bool all_threads_idle() {
    bool all_idle = true;
    omp_set_lock(&lock);
    for (int i = 0; i < num_threads; i++) {
      if (thread_is_working[i]) {
        all_idle = false;
        break;
      }
    }
    omp_unset_lock(&lock);
    return all_idle;
  }

  bool thread_working(int tid) {
    bool th_is_working;
    omp_set_lock(&lock);
    th_is_working = thread_is_working[tid];
    omp_unset_lock(&lock);
    return th_is_working;
  }

  void thread_process_received_data(int tid, Thread_private_data &tpd) {
  }

  bool can_thread_split_work(int tid, Thread_private_data &tpd) {
    if (!thread_working(tid)) return false;
    tpd.task_split_level = 0;
    // start search from level 0 task queue
    while (tpd.task_split_level < tpd.depth && tpd.sizes[tpd.task_split_level] - tpd.idx[tpd.task_split_level] < task_split_threshold)
      tpd.task_split_level++;
    if (tpd.sizes.size() > tpd.task_split_level && tpd.sizes[tpd.task_split_level] - tpd.idx[tpd.task_split_level] >= task_split_threshold)
      return true;
    return false;
  }

  void thread_split_work(int stealer, int &num_tasks_to_steal, int victim, std::vector<Thread_private_data> &tpd) {
    auto l = tpd[victim].task_split_level;
    auto remaining_tasks = tpd[victim].sizes[l] - tpd[victim].idx[l];
    num_tasks_to_steal = remaining_tasks / 2;
    remaining_tasks -= num_tasks_to_steal;
    tpd[stealer].depth = l;
    tpd[stealer].sizes[l] = tpd[victim].sizes[l];
    tpd[stealer].idx[l] = tpd[stealer].sizes[l] - num_tasks_to_steal;
    tpd[victim].sizes[l] = tpd[victim].idx[l] + remaining_tasks;
  }

  bool receive_data(int source, int size, int tid, Thread_private_data &tpd) {
    if(size == 0) {
      if(scheme == RP) {
        next_work_request[tid] = random() % num_threads;
        while(next_work_request[tid] == tid)
          next_work_request[tid] = random() % num_threads;
      } else { //ARR
        next_work_request[tid] = (next_work_request[tid] + 1) % num_threads;
        if(next_work_request[tid] == tid) //make sure that the request is not made to self
          next_work_request[tid] = (next_work_request[tid] + 1) % num_threads;
      }
      requested_work_from[tid].erase(requested_work_from[tid].begin());
      return false;
    }
    if(requested_work_from[tid].size() != 1 || requested_work_from[tid][0] != source ) {
      exit(1);
    }
    // nothing else to do, data is already pushed in the queue by the donor thread
    // process the data put in the shared queue
    thread_process_received_data(tid, tpd);

    if(scheme == RP) {
      next_work_request[tid] = random() % num_threads;
      while(next_work_request[tid] == tid)
        next_work_request[tid] = random() % num_threads;
    } else { //ARR
      next_work_request[tid] = (next_work_request[tid] + 1) % num_threads;
      if(next_work_request[tid] == tid) //make sure that the request is not made to self
        next_work_request[tid] = (next_work_request[tid] + 1) % num_threads;
    }
    requested_work_from[tid].erase(requested_work_from[tid].begin());
    thread_start_working(tid);
    return true;
  }

  void process_work_split_request(int source, int tid, std::vector<Thread_private_data> &tpd) {
    // if thread_id = 0 and all threads are idle, no need to send response now
    // if global work is received the response will be sent eventually
    if (tid == 0 && all_threads_idle()) {
      all_idle_work_request_counter++;
      return;
    }
    if (!thread_working(tid) || !can_thread_split_work(tid, tpd[tid])) {
      int buffer[2];
      buffer[0] = RT_WORK_RESPONSE;
      buffer[1] = 0;
      send_msg(buffer, 2, tid, source);
      return;
    }
    int length = 0;
    thread_split_work(source, length, tid, tpd);
    int buffer_size[2];
    buffer_size[0] = RT_WORK_RESPONSE;
    buffer_size[1] = length; // put there length of the split stack split.size()+1;
    send_msg(buffer_size, 2, tid, source);
  }

  void send_msg(int *buffer, int length, int src_thr, int dest_thr) {
    omp_set_lock(&lock);
    message_queue[dest_thr].push_back(src_thr);
    for(int i = 0; i <length; i++)
      message_queue[dest_thr].push_back(buffer[i]);
    omp_unset_lock(&lock);
  }

  void recv_msg(int *buffer, int length, int thr, int originating_thr) {
    omp_set_lock(&lock);
    int source = message_queue[thr].front();
    if(originating_thr != source) {
      exit(0);
    }
    message_queue[thr].pop_front(); //take off source
    for(int i = 0; i < length; i++) {
      buffer[i] = message_queue[thr].front();
      message_queue[thr].pop_front();
    }
    omp_unset_lock(&lock);
  }

  int check_request(int tid) {
    if(message_queue[tid].size() > 0 ) {
      omp_set_lock(&lock);
      int source = message_queue[tid].front();
      omp_unset_lock(&lock);
      return source;
    }
    return -1;
  }

  void process_request(int source, int tid, std::vector<Thread_private_data> &tpd) {
    int recv_buf[2];
    recv_msg(recv_buf, 2, tid, source);
    switch(recv_buf[0]) {
      case RT_WORK_REQUEST:
        process_work_split_request(source, tid, tpd);
        break;
      case RT_WORK_RESPONSE:
        receive_data(source, recv_buf[1], tid, tpd[tid]);
        return;
      default: exit(1);
    }
  }

  void send_work_request(int tid) {
    if (!requested_work_from[tid].empty()) return;
    int buffer[2];
    buffer[0] = RT_WORK_REQUEST;
    buffer[1] = 0;       // filler
    send_msg(buffer, 2, tid, next_work_request[tid]);
    requested_work_from[tid].push_back(next_work_request[tid]);
  }

  void task_schedule(int tid, std::vector<Thread_private_data> &tpd) {
    /*
    int src = check_request(tid);
    if (src != -1) process_request(src, tid, tpd);
    if (!thread_working(tid) && !all_threads_idle())
      send_work_request(tid); // if idle, ask for task stealing
    //*/
  }
};
