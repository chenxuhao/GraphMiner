#pragma once

#ifdef USE_NUMA
  #include <numa.h>
  template<typename T>
  T* custom_alloc_local(size_t elements) {
    return (T*)numa_alloc_local(sizeof(T) * elements);
    //return (T*)aligned_alloc(PAGE_SIZE, elements * sizeof(T));
  }
  template<typename T>
  T* custom_alloc_global(size_t elements) {
    return (T*)numa_alloc_interleaved(sizeof(T) * elements);
  }
  template<typename T>
  void custom_free(T *ptr, size_t elements) {
    numa_free(ptr, sizeof(T)*elements);
  }
#else
  template<typename T>
  T* custom_alloc_local(size_t elements) {
    return new T[elements];
  }
  template<typename T>
  T* custom_alloc_global(size_t elements) {
    return new T[elements];
  }
  template<typename T>
  void custom_free(T *ptr, size_t elements) {
    delete[] ptr;
  }
#endif

