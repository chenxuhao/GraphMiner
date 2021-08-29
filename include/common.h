#pragma once
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <set>
#include <map>
#include <deque>
#include <vector>
#include <limits>
#include <cstdio>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <climits>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

typedef uint8_t BYTE;
typedef uint8_t mask_t;
typedef uint8_t label_t;
typedef uint8_t vlabel_t;
typedef uint8_t elabel_t;
typedef uint8_t cmap_vt; // cmap value type
typedef int32_t VertexId;
typedef int32_t VertexID;
typedef int64_t EdgeID;
typedef int32_t IndexT;
typedef int32_t WeightT;
typedef std::vector<VertexId> VertexList;
typedef std::vector<std::vector<VertexId>> VertexLists;
typedef std::vector<BYTE> ByteList;
typedef unsigned long long AccType;
typedef uint64_t emb_index_t;
typedef int32_t vidType;
typedef int64_t eidType;

#define ADJ_SIZE_THREASHOLD 1024
#define FULL_MASK 0xffffffff
#define MAX_PATTERN_SIZE 8
#define NUM_BUCKETS 128
#define BUCKET_SIZE 1024
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define BLOCK_SIZE    256
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define MAX_THREADS (30 * 1024)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define MAX_BLOCKS (MAX_THREADS / BLOCK_SIZE)
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

enum Status {
  Idle,
  Extending,
  IteratingEdge,
  Working,
  ReWorking
};

#define OP_INTERSECT 'i'
#define OP_DIFFERENCE 'd'
extern std::map<char,double> time_ops;
