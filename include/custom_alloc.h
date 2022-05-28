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

template<typename T>
static void read_file(std::string fname, T *& pointer, size_t length) {
  pointer = custom_alloc_global<T>(length);
  assert(pointer);
  std::ifstream inf(fname.c_str(), std::ios::binary);
  if (!inf.good()) {
    std::cerr << "Failed to open file: " << fname << "\n";
    exit(1);
  }
  inf.read(reinterpret_cast<char*>(pointer), sizeof(T) * length);
  inf.close();
}

template<typename T>
static void map_file(std::string fname, T *& pointer, size_t length) {
  int inf = open(fname.c_str(), O_RDONLY, 0);
  if (-1 == inf) {
    std::cerr << "Failed to open file: " << fname << "\n";
    exit(1);
  }
  pointer = (T*)mmap(nullptr, sizeof(T) * length, PROT_READ, MAP_SHARED, inf, 0);
  assert(pointer != MAP_FAILED);
  close(inf);
}

