#pragma once
#include <cusparse_v2.h>

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                      \
    cudaError err = call;                                                      \
    if( cudaSuccess != err) {                                                  \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n",\
                err, __FILE__, __LINE__, cudaGetErrorString( err) );           \
        exit(EXIT_FAILURE);                                                    \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);              \

#  define CUDA_SAFE_THREAD_SYNC( ) {                                           \
    cudaError err = CUT_DEVICE_SYNCHRONIZE();                                  \
    if ( cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
                __FILE__, __LINE__, cudaGetErrorString( err) );                \
    } }

#if __CUDACC_VER_MAJOR__ >= 9
#define SHFL_DOWN(a,b) __shfl_down_sync(0xFFFFFFFF,a,b)
#define SHFL(a,b) __shfl_sync(0xFFFFFFFF,a,b)
#else
#define SHFL_DOWN(a,b) __shfl_down(a,b)
#define SHFL(a,b) __shfl(a,b)
#endif

template <typename T>
__forceinline__ __device__ T warp_reduce(T val) {
  T sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8);
  sum += SHFL_DOWN(sum, 4);
  sum += SHFL_DOWN(sum, 2);
  sum += SHFL_DOWN(sum, 1);
  sum  = SHFL(sum, 0);
  return sum;
}

__forceinline__ __device__ void warp_reduce_iterative(vidType &val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  val  = SHFL(val, 0);
}

// from http://forums.nvidia.com/index.php?showtopic=186669
static __device__ unsigned get_smid(void) {
  unsigned ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

#define VLIST_CACHE_SIZE 256
__forceinline__ __device__ void warp_load_mem_to_shm(vidType* from, vidType* to, vidType len) {
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  for (vidType id = thread_lane; id < len; id += WARP_SIZE) {
    to[id] = from[id];
  }
  __syncwarp();
}

// you must first call the cudaGetDeviceProperties() function, then pass
// the devProp structure returned to this function:
int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major){
    case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
    case 3: // Kepler
      cores = mp * 192;
      break;
    case 5: // Maxwell
      cores = mp * 128;
      break;
    case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
    default:
      printf("Unknown device type\n");
      break;
  }
  return cores;
}

static size_t print_device_info(bool print_all) {
  int deviceCount = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
  printf("Found %d devices\n", deviceCount);
  // Another way to get the # of cores: #include <helper_cuda.h> in this link:
  // https://github.com/NVIDIA/cuda-samples/blob/6be514679b201c8a0f0cda050bc7c01c8cda32ec/Common/helper_cuda.h
  //int CUDACores = _ConvertSMVer2Cores(props.major, props.minor) * props.multiProcessorCount;
  size_t mem_size = 0;
  for (int device = 0; device < deviceCount; device++) {
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaSetDevice(device));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
    printf("  Device[%d]: %s\n", device, prop.name);
    if (device == 0 || print_all) {
      printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
      printf("  Warp size: %d\n", prop.warpSize);
      printf("  Total # SM: %d\n", prop.multiProcessorCount);
      printf("  Total # CUDA cores: %d\n", getSPcores(prop));
      printf("  Total amount of shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
      printf("  Total # registers per block: %d\n", prop.regsPerBlock);
      printf("  Total amount of constant memory: %lu bytes\n", prop.totalConstMem);
      printf("  Total global memory: %.1f GB\n", float(prop.totalGlobalMem)/float(1024*1024*1024));
      printf("  Memory Clock Rate: %.2f GHz\n", float(prop.memoryClockRate)/float(1024*1024));
      printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
      //printf("  Maximum memory pitch: %u\n", prop.memPitch);
      printf("  Peak Memory Bandwidth: %.2f GB/s\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
      mem_size = prop.totalGlobalMem;
    }
  }
  return mem_size;
}

static size_t get_gpu_mem_size(int device = 0) {
  cudaDeviceProp prop;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
  return prop.totalGlobalMem;
}

