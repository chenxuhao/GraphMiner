// Copyright (c) 2020 MIT
#pragma once
#include <sys/time.h>
#include <iostream>

class Timer {
public:
  Timer() : Timer("null") {}
  Timer(std::string name) : name_(name) {}
  void Start() {
    gettimeofday(&start_time_, NULL);
  }
  void Stop() {
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
  }
  double Seconds() const {
    return (double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1e6;
  }
  double Millisecs() const {
    return (double)1000*(double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/(double)1000;
  }
  double Microsecs() const {
    return 1e6*(double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec;
  }
  std::string get_name() { return name_; }
private:
  std::string name_;
  struct timeval start_time_;
  struct timeval elapsed_time_;
};

// Times op's execution using the timer t
#define TIME_OP(t, op) { t.Start(); (op); t.Stop(); }

template <typename F>
void timeThis(const F& f, const std::string name) {
  Timer t(name);
  t.Start();
  f();
  t.Stop();
  std::cout << "runtime[" << t.get_name() << "] = " << t.Seconds() << " sec\n";
}

inline uint64_t read_cycle() {
  volatile uint64_t t = 0;
#ifdef PROFILE_LATENCY
  asm __volatile__(
      "lfence\n"
      // Guaranteed to clear the high-order 32 bits of RAX and RDX.
      "rdtsc\n"
      "shlq $32, %%rdx\n"
      "orq %%rdx, %%rax\n"
      : "=a" (t)
      :
      : "%rdx"
      );
#endif
  return t;
}

