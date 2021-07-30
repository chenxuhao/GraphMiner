#include <omp.h>
#include <vector>

template <typename InTy = unsigned, typename OutTy = unsigned>
inline void parallel_prefix_sum(const std::vector<InTy>& in, OutTy *prefix) {
  const size_t block_size = 1 << 20;
  const size_t num_blocks = (in.size() + block_size - 1) / block_size;
  std::vector<OutTy> local_sums(num_blocks);
  // count how many bits are set on each thread
  #pragma omp parallel for
  for (size_t block = 0; block < num_blocks; block ++) {
    OutTy lsum       = 0;
    size_t block_end = std::min((block + 1) * block_size, in.size());
    for (size_t i = block * block_size; i < block_end; i++)
      lsum += in[i];
    local_sums[block] = lsum;
  }
  std::vector<OutTy> bulk_prefix(num_blocks + 1);
  OutTy total = 0;
  for (size_t block = 0; block < num_blocks; block++) {
    bulk_prefix[block] = total;
    total += local_sums[block];
  }
  bulk_prefix[num_blocks] = total;
  #pragma omp parallel for
  for (size_t block = 0; block < num_blocks; block ++) {
    OutTy local_total = bulk_prefix[block];
    size_t block_end  = std::min((block + 1) * block_size, in.size());
    for (size_t i = block * block_size; i < block_end; i++) {
      prefix[i] = local_total;
      local_total += in[i];
    }
  }
  prefix[in.size()] = bulk_prefix[num_blocks];
}

template <typename InTy = unsigned, typename OutTy = unsigned>
inline void prefix_sum(const std::vector<InTy>& in, OutTy *prefix) {
  OutTy total = 0;
  for (size_t n = 0; n < in.size(); n++) {
    prefix[n] = total;
    total += (OutTy)in[n];
  }
  prefix[in.size()] = total;
}

