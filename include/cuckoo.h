// Cuckoo hashing.
#include "defines.h"
#include "common.h"
#include <bits/stdc++.h>
#include <functional>
#include <vector>
#include <utility>

//#define ENABLE_CUCKOO_PROFILE
//#define USE_PRIME
//#define ENABLE_INLINE
#define NULL_VALUE 0
#define NUM_MERSENNE 9
//int MERSENNE_PRIME[NUM_MERSENNE] = { 127, 509, 1021, 2053, 4093, 8191, 16381, 32771}; //131071, 524287};
int MERSENNE_PRIME[NUM_MERSENNE] = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
int MERSENNE_PRIME_EXP[NUM_MERSENNE] = { 7, 13, 17, 19, 23, 29, 31, 37, 41};

#if 1
#ifdef ENABLE_INLINE
template <typename T>
inline T mod(T key, int r, T table_size, T shift) {
#else
template <typename T>
__attribute__((noinline)) T mod(T key, int r, T table_size, T shift) {
#endif
  T x = T(key * r);
  //x = (x & table_size) + (x >> shift);
  //return x % table_size;
  return x & (table_size-1);
}
#else
#ifdef ENABLE_INLINE
template <typename T>
inline T mod(T key, int r, T table_size, T shift) {
#else
template <typename T>
__attribute__((noinline)) T mod(T key, int r, T table_size, T shift) {
#endif
  T x = T(key * r);
  x = (x & table_size) + (x >> shift);
  return (x >= table_size) ? (x % table_size) : x;
  //return x ? x & table_size - 1 : x;
}
#endif

inline int get_shift(int table_size) {
  return 32 - (int)(log(table_size) / log(2) + 0.5);
}

template <typename KeyT, typename ValueT>
class HashEntry {
public:
  KeyT key;
  ValueT value;
  HashEntry() : HashEntry(0, NULL_VALUE) {}
  HashEntry(KeyT k, ValueT v) :
    key(k), value(v) {};
  inline bool is_null() { return value == NULL_VALUE; }
  void reset() { value = NULL_VALUE; }
  HashEntry& operator=(const HashEntry &other) {
    key = other.key;
    value = other.value;
    return *this;
  }
  bool operator==(const HashEntry &other) {
    if (key == other.key && value == other.value) return true;
    return false;
  }
};

// max_size: upper bound on number of elements in our set.
// num_tables: number of tables, i.e., number of choices for position.
// max_cycle: max number of times to declare a cycle;
// rehashing the entire map if a cycle found (deadlock).
template <typename KT, typename VT, int num_tables=2>
class cuckoo_map {
private:
  std::vector<std::vector<HashEntry<KT, VT>>> hashtable;
  // Array to store possible positions for a key
  VT temp_value;
  const float MAX_LOAD = 0.4f;
  int a1; // random number for hash function 1
  int a2; // random number for hash function 2
  int mersenne_index;
  int max_loop; // maximum number of loops to insert a new entry
  int num_entries; // number of entries inserted into the tables
  int num_buckets; // number of slots available in the table
  int num_bits;
  int num_rehash;
  int num_queries;
  int num_h1_queries;
  int num_h2_queries;
  int num_null_queries;
  int num_insertion;
  int num_replacement;
  KT max_degree;

  // function to fill hash table with init value 0
  void clean_table() {
    for (int i = 0; i < num_tables; i++) {
      for (int j = 0; j < num_buckets; j++) {
        hashtable[i][j].reset();
      }
    }
  }
  void rehash() {
    std::vector<HashEntry<KT,VT>> table;
    // collect all valid entries
    for (int i = 0; i < num_tables; ++i) {
      for (int j = 0; j < num_buckets; ++j) {
        if (!hashtable[i][j].is_null()) {
          table.push_back(hashtable[i][j]);
          hashtable[i][j].reset();
        }
      }
    }
    //if (table.size() != size_t(num_entries)) {
    //  std::cout << "num_entries = " << num_entries 
    //            << " table.size = " << table.size() << "\n";
    //  exit(1);
    //}
    //assert(KT(num_entries) <= max_degree);
    if (float(num_entries)/float(num_buckets) > MAX_LOAD)
      grow();
    a1 = rand() % (num_buckets - 1) + 1;
    a2 = rand() % (num_buckets - 1) + 1;
    //std::cout << "a1: " << a1 << " a2: " << a2 << " num_entries: " << num_entries << " num_buckets: " << num_buckets << std::endl;
    num_entries = 0;
    for (size_t i = 0; i < table.size(); ++i)
      insert(table[i]);
  }
  void grow() {
    ++ mersenne_index;
    if (mersenne_index >= NUM_MERSENNE) {
      std::cout << "ERROR: mersenne overflow\n";
      exit(1);
    }
    int old_buckets = num_buckets;
    num_buckets *= 2;
#ifdef USE_PRIME
    num_buckets = MERSENNE_PRIME[mersenne_index];
    num_bits = MERSENNE_PRIME_EXP[mersenne_index];
#else
    num_buckets *= 2;
    num_bits = get_shift(num_buckets);
#endif
    //max_loop = num_buckets/2;
    for (int i = 0; i < num_tables; ++i) {
      hashtable[i].resize(num_buckets);
      for (int j = old_buckets; j < num_buckets; ++j)
        hashtable[i][j].reset();
    }
  }
#ifdef ENABLE_INLINE
  KT hash(int table_id, KT key) {
#else
  __attribute__((noinline)) KT hash(int table_id, KT key) {
#endif
    // if (table_id == 1)
    //   return hash1(key);
    //   //return ThomasWangHash(key) % num_buckets;
    // if (table_id == 2)
    //   return hash2(key);
    //   //return BobJenkinsIntHash(key) % num_buckets;
    
    // num_tables is compile-time constant
    if (table_id == 1 && 1 <= num_tables) return hash1(key);
    if (table_id == 2 && 2 <= num_tables) return hash2(key);
    if (table_id == 3 && 3 <= num_tables) return hash3(key);
    if (table_id == 4 && 4 <= num_tables) return hash4(key);
    std::cout << "Error: table_id not supported!\n";
    return 0;
  }

#ifdef ENABLE_INLINE
  KT hash1(KT key) {
#else
  __attribute__((noinline)) KT hash1(KT key) {
#endif
    return mod<KT>(key, a1, num_buckets, num_bits);
  }
#ifdef ENABLE_INLINE
  KT hash2(KT key) {
#else
  __attribute__((noinline)) KT hash2(KT key) {
#endif
    return mod<KT>(key, a2, num_buckets, num_bits);
  }
  KT hash3(KT key) {
    return (key*a1) % num_buckets;
  }
  KT hash4(KT key) {
    return (key*a2/num_buckets) % num_buckets;
  }
  uint32_t ThomasWangHash(uint32_t key) {
    //http://burtleburtle.net/bob/hash/integer.html
    key += ~(key << 15);
    key ^= (key >> 10);
    key += (key << 3);
    key ^= (key >> 6);
    key += ~(key << 11);
    key ^= (key >> 16);
    return key;
  }
  uint32_t BobJenkinsIntHash(uint32_t key) {
    //https://gist.github.com/badboy/6267743
    key = (key+0x7ed55d16) + (key<<12);
    key = (key^0xc761c23c) ^ (key>>19);
    key = (key+0x165667b1) + (key<<5);
    key = (key+0xd3a2646c) ^ (key<<9);
    key = (key+0xfd7046c5) + (key<<3);
    key = (key^0xb55a4f09) ^ (key>>16);
    return key;
  }
  std::function<KT(KT)> hashfunc[4];
public:
  cuckoo_map() : cuckoo_map(1024) { }
  cuckoo_map(uint32_t max) :
    hashfunc{ [this](KT k) { return this->hash1(k); },
              [this](KT k) { return this->hash2(k); },
              [this](KT k) { return this->hash3(k); },
              [this](KT k) { return this->hash4(k); }
    }
  { init(max); }
  ~cuckoo_map() {}
  int get_num_buckets() const { return num_buckets; }
  int get_num_rehash() const { return num_rehash; }
  int get_num_queries() const { return num_queries; }
  int get_num_h1_queries() const { return num_h1_queries; }
  int get_num_h2_queries() const { return num_h2_queries; }
  int get_num_misses() const { return num_null_queries; }
  int get_num_insertion() const { return num_insertion; }
  int get_num_replacement() const { return num_replacement; }
  int get_num_grow() const { return mersenne_index; }
  void clear() { num_entries = 0; }
  //void clear() { }
  void init(KT n) {
    max_degree = n;
    num_entries = 0;
    mersenne_index = 0;
    num_rehash = 0;
    num_queries = 0;
    num_h1_queries = 0;
    num_h2_queries = 0;
    num_null_queries = 0;
    num_insertion = 0;
    num_replacement = 0;
#ifdef USE_PRIME
    num_buckets = MERSENNE_PRIME[mersenne_index];
    num_bits = MERSENNE_PRIME_EXP[mersenne_index];
#else
    num_buckets = n;
    num_bits = get_shift(num_buckets);
#endif
    a1 = rand() % (num_buckets-1) + 1;
    a2 = rand() % (num_buckets-1) + 1;
    //max_loop = num_buckets;
    //max_loop = 4 + (int)(4 * log(num_buckets) / log(2) + 0.5);
    max_loop = num_buckets/2;
    hashtable.resize(num_tables);
    for (int i = 0; i < num_tables; i++)
      hashtable[i].resize(num_buckets);
    clean_table();
  }

  // print hash table contents
  void printTable() {
    printf("hash tables:\n");
    for (int i = 0; i < num_tables; i++) {
      printf("\n");
      for (KT j = 0; j < num_buckets; j++) {
        (hashtable[i][j].is_null()) ? printf("- ") :
          printf("(%d, %d) ", hashtable[i][j].key, hashtable[i][j].value);
      }
    }
    printf("\n");
  }
  // get the value using the key
#ifdef ENABLE_INLINE
  VT get(KT k) {
#else
  __attribute__((noinline)) VT get(KT k) {
#endif
#ifdef ENABLE_CUCKOO_PROFILE
    num_queries ++;
#endif
    for (int i = 0; i < num_tables; i++) { 
      auto pos = hash(i+1, k);
      if (hashtable[i][pos].key == k) {
#ifdef ENABLE_CUCKOO_PROFILE
        if (i == 0) num_h1_queries ++;
        else num_h2_queries ++;
#endif
        return hashtable[i][pos].value;
      }
    } 
#ifdef ENABLE_CUCKOO_PROFILE
    num_null_queries ++;
#endif
    return NULL_VALUE;
  }
  // set value using the key
  // if key not exist, insert a new entry
#ifdef ENABLE_INLINE
  void set(KT k, VT v) {
#else
  __attribute__((noinline)) void set(KT k, VT v) {
#endif
    for (int i = 0; i < num_tables; i++) { 
      auto pos = hash(i+1, k);
      if (hashtable[i][pos].key == k) {
        auto &val = hashtable[i][pos].value;
        if (v != val) {
          if (val == NULL_VALUE) num_entries ++;
          else if (v == NULL_VALUE) num_entries --;
          val = v;
        }
        return;
      }
    }
    if (v != NULL_VALUE) insert(HashEntry(k,v));
  }
  // insert a new entry
  void insert(HashEntry<KT,VT> entry) {
#ifdef ENABLE_CUCKOO_PROFILE
    num_insertion ++;
#endif
    //if (find(entry.key)) return;
    for (int i = 0; i < max_loop; ++i) {
      for (int j = 0; j < num_tables; ++j) {
        auto pos = hash(j + 1, entry.key);
        assert(pos < KT(num_buckets));
        std::swap(entry, hashtable[j][pos]);
        if (entry.is_null()) { ++num_entries; return; }
#ifdef ENABLE_CUCKOO_PROFILE
        num_replacement ++;
#endif
      }
    }
#ifdef ENABLE_CUCKOO_PROFILE
    num_rehash ++;
#endif
    rehash();
    insert(entry);
  }
  // find the entry using the key
  bool find(KT key) {
    for (int i = 0; i < num_tables; i++) {
      auto pos = hash(i+1, key);
      if (hashtable[i][pos].key == key &&
          !hashtable[i][pos].is_null()) {
        return true;
      }
    }
    return false;
  }
};

static inline void print_cuckoo_stats(std::vector<cuckoo_map<uint32_t,uint8_t>> cmaps) {
  int num_threads = cmaps.size();
  int max_num_buckets = 0;
  for (int tid = 0; tid < num_threads; tid ++) {
    if (max_num_buckets < cmaps[tid].get_num_buckets())
      max_num_buckets = cmaps[tid].get_num_buckets();
  }
#ifdef ENABLE_CUCKOO_PROFILE
  int num_grow = 0;
  int num_rehash = 0;
  int64_t num_queries = 0;
  int64_t num_h1_queries = 0;
  int64_t num_h2_queries = 0;
  int64_t num_misses = 0;
  int num_insertion = 0;
  int num_replacement = 0;
  for (int tid = 0; tid < num_threads; tid ++) {
    num_grow += cmaps[tid].get_num_grow();
    num_rehash += cmaps[tid].get_num_rehash();
    num_queries += cmaps[tid].get_num_queries();
    num_h1_queries += cmaps[tid].get_num_h1_queries();
    num_h2_queries += cmaps[tid].get_num_h2_queries();
    num_misses += cmaps[tid].get_num_misses();
    num_insertion += cmaps[tid].get_num_insertion();
    num_replacement += cmaps[tid].get_num_replacement();
  }
  std::cout << "Number of grow: " << num_grow << "\n";
  std::cout << "Number of rehash: " << num_rehash << "\n";
  std::cout << "Number of queries: " << num_queries << "\n";
  std::cout << "Number of h1 queries: " << num_h1_queries << "\n";
  std::cout << "Number of h2 queries: " << num_h2_queries << "\n";
  std::cout << "Number of misses: " << num_misses << "\n";
  std::cout << "Number of insertion: " << num_insertion << "\n";
  std::cout << "Number of replacement: " << num_replacement << "\n";
  std::cout << "Hash table miss rate: " << float(num_misses) / float(num_queries) << "\n";
#endif
  std::cout << "Max number of buckets: " << max_num_buckets << "\n";
}

