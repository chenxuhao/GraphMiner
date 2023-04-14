#export LD_LIBRARY_PATH=/usr/local/OpenBLAS/build/lib:$LD_LIBRARY_PATH
#export KMP_AFFINITY=scatter
export KMP_AFFINITY=disabled
export KMP_LIBRARY=turnaround
export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=20

export CUDA_HOME=/usr/local/cuda
#CUDA_HOME=/org/centers/cdgc/cuda/cuda-10.2
#CUDA_HOME=/jet/packages/spack/opt/spack/linux-centos8-zen/gcc-8.3.1/cuda-10.2.89-kz7u4ix6ed53nioz4ycqin3kujcim3bs
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_MPI_SUPPORT=1
#export NVSHMEM_SHMEM_SUPPORT=1
export MPI_HOME=/usr
export NVSHMEM_PREFIX=/usr/local/nvshmem
export NVSHMEM_HOME=/usr/local/nvshmem
#PAPI_HOME = /usr/local/papi-6.0.0
#ICC_HOME = /opt/intel/compilers_and_libraries/linux/bin/intel64
export OPENBLAS_DIR=/usr/local/openblas
#OPENBLAS_DIR=/org/centers/cdgc/openblas/ubuntu-gcc7.5
#OPENBLAS_DIR=/org/centers/cdgc/openblas/centos-gcc9.2
#OPENBLAS_DIR=/ocean/projects/cie170003p/shared/OpenBLAS/build
export MKL_DIR=/opt/apps/sysnet/intel/20.0/mkl
#export MKLROOT=/opt/intel/mkl

export GCC_HOME=/usr/lib/gcc/x86_64-linux-gnu/8
export TBB_HOME=/opt/intel/oneapi/tbb/latest
# export CILK_HOME=/home/cxh/OpenCilk/build
# export CILK_CLANG=/home/cxh/OpenCilk/build/lib/clang/14.0.6
export CILK_HOME=/home/azliu/build
export CILK_CLANG=/home/azliu/build/lib/clang/14.0.6

export PAPI_HOME=/usr/local/papi-6.0.0
#export CUB_DIR=../../cub
#export MGPU_DIR=../../moderngpu

