#pragma once

// defined by default: ENABLE_INLINE, USE_DAG == 1, USE_CMAP, IDENT_CMAP
// not defined by default: DISABLE_INLINE, ENABLE_CUCKOO_PROFILE, USE_MERGE

// inline critical functions
#ifndef DISABLE_INLINE
#define ENABLE_INLINE
#endif

// input graph 
#ifndef USE_DAG
#define USE_DAG 1
#endif

// set operation implementation
#if !defined(USE_CMAP) && !defined(USE_MERGE)
#define USE_CMAP
#endif

// hash table implementation
#if !defined(CUCKOO_CMAP) && !defined(STL_CMAP) && !defined(IDENT_CMAP)
#define IDENT_CMAP
#endif

#ifdef CUCKOO_CMAP
  #undef STL_CMAP
  #undef IDENT_CMAP

  #ifndef CUCKOO_WAY
  #define CUCKOO_WAY 2
  #endif
#endif

#ifdef STL_CMAP
  #undef CUCKOO_CMAP
  #undef IDENT_CMAP
#endif

#ifdef IDENT_CMAP
  #undef CUCKOO_CMAP
  #undef STL_CMAP
#endif

// #define ENABLE_CUCKOO_PROFILE

// TODO: merge join implementation
#if !defined(BRANCH_MERGE) && !defined(BLESS_MERGE) && !defined(SIMD_MERGE)
#define BRANCH_MERGE
#endif

// profiler
//#define ENABLE_PAPI
