#include <stdio.h>
#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <cstring>
#include <unistd.h>
#include <climits>
#include <omp.h>

using namespace std;

#define NULL_VAL 2147483647
#define CREATE_SIZE 100
#define DEBUG 0
#define COLLECT_DATA 0
#define TIME_INNERLOOP 0
#ifndef SIMPLE
#define SIMPLE 0
#else
#undef SIMPLE
#define SIMPLE 1
#endif

#include "graph.h"
#include "pattern.hpp"
#include "util.hpp"
#include "output.hpp"
#include "dynamic_table.hpp"
#include "dynamic_table_array.hpp"
#include "partitioner.hpp"
#if SIMPLE
  #include "colorcount_simple.hpp"
#else
  #include "colorcount.hpp"
#endif

bool timing = false;

void print_info_short(char* name);
void print_info(char* name);
void count(Graph &g, char* template_file, bool labeled, bool do_vert, bool do_gdd,
           int iterations, bool do_outerloop, bool calc_auto);
 
int main(int argc, char** argv) {
  // remove buffer so all outputs show up before crash
  setbuf(stdout, NULL);
  char* graph_file = NULL;
  char* template_file = NULL;
  char* batch_file = NULL;
  int iterations = 1;
  bool do_outerloop = false;
  bool calculate_automorphism = true;
  bool labeled = false;
  bool do_gdd = false;
  bool do_vert = false;
  bool verbose = false;
  int motif = 0;

  char c;
  while ((c = getopt (argc, argv, "g:t:b:i:m:acdvrohl")) != -1) {
    switch (c) {
      case 'h':
        print_info(argv[0]);
        break;
      case 'l':
        labeled = true;
        break;
      case 'g':
        graph_file = strdup(optarg);
        break;
      case 't':
        template_file = strdup(optarg);
        break;
      case 'b':
        batch_file = strdup(optarg);
        break;
      case 'i':
        iterations = atoi(optarg);
        break;
      case 'm':
        motif = atoi(optarg);
        break;
      case 'a':
        calculate_automorphism = false; 
        break;
      case 'c':
        do_vert = true;
        break;
      case 'd':
        do_gdd = true;
        break;
      case 'o':
        do_outerloop = true;
        break;
      case 'v':
        verbose = true;
        break;
      case 'r':
        timing = true;
        break;
      case '?':
        if (optopt == 'g' || optopt == 't' || optopt == 'b' || optopt == 'i' || optopt == 'm')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n",
      optopt);
        print_info(argv[0]);
      default:
        abort();
    }
  } 

  if(argc < 3)
    print_info_short(argv[0]);

  if (motif && (motif < 3 || motif > 10)) {
    printf("\nMotif option must be between [3,10]\n");    
    print_info(argv[0]);
  } else if (graph_file == NULL) { 
    printf("\nMust supply graph file\n");    
    print_info(argv[0]);
  } else if (template_file == NULL && batch_file == NULL && !motif) {
    printf("\nMust supply template XOR batchfile or -m option\n");
    print_info(argv[0]);
  } else if (template_file != NULL && batch_file != NULL) {
    printf("\nMust only supply template file XOR batch file\n");
    print_info(argv[0]);
  } else if (iterations < 1) {
    printf("\nNumber of iterations must be positive\n");    
    print_info(argv[0]);
  }

  Graph g(graph_file);
  assert(template_file != NULL);
  Timer t;
  t.Start();
  count(g, template_file, labeled, do_vert, do_gdd,
        iterations, do_outerloop, calculate_automorphism);
  t.Stop();

  return 0;
}

void print_info_short(char* name) {
  printf("\nTo run: %s [-g graphfile] [-t template]\n", name);
  printf("Example: ./colorcoding_omp_base -g ~/data/citeseer/graph  -t 5-path.graph\n");
  //printf("Help: %s -h\n\n", name);
  exit(0);
}

void print_info(char* name) {
  /*
  printf("\nTo run: %s [-g graphfile] [-t template || -b batchfile] [options]\n\n", name);

  printf("\tgraphfile = \n");
  printf("\t\tn\n");
  printf("\t\tm\n");
  printf("\t\tv0 v1\n");
  printf("\t\tv0 v2\n");
  printf("\t\t...\n");
  printf("\t\t(zero indexed)\n\n");

  printf("\tgraphfile (if labeled) = \n");
  printf("\t\tn\n");
  printf("\t\tm\n");
  printf("\t\tlabel_v0\n");
  printf("\t\tlabel_v1\n");
  printf("\t\t...\n");
  printf("\t\tv0 v1\n");
  printf("\t\tv0 v2\n");
  printf("\t\t...\n");
  printf("\t\t(zero indexed)\n\n"); 

  printf("\ttemplate =\n");
  printf("\t\tsame format as graphfile\n\n");

  printf("\tbatchfile =\n");
  printf("\t\ttemplate1\n");
  printf("\t\ttemplate2\n");
  printf("\t\t...\n");
  printf("\t\t(must supply only one of template file or batchfile)\n\n");

  printf("\toptions = \n");
  printf("\t\t-m  [#], compute counts for motifs of size #\n");
  printf("\t\t-o  Use outerloop parallelization\n");
  printf("\t\t-l  Graph and template are labeled\n");
  printf("\t\t-i  [# iterations], default: 1\n");
  printf("\t\t-c  Output per-vertex counts to [template].vert\n");
  printf("\t\t-d  Output graphlet degree distribution to [template].gdd\n");
  printf("\t\t-a  Do not calculate automorphism of template\n");
  printf("\t\t\t(recommended when template size > 10)\n");
  printf("\t\t-r  Report runtime\n");
  printf("\t\t-v  Verbose output\n");
  printf("\t\t-h  Print this\n\n");
  */
  exit(0);
}

void count(Graph &g, char* template_file, bool labeled, bool do_vert, bool do_gdd,
           int iterations, bool do_outerloop, bool calc_auto) {
  int* srcs_g;
  int* dsts_g;
  int* labels_g;
  int* srcs_t;
  int* dsts_t;
  int* labels_t;
  bool verbose = false;

  char* vert_file = new char[1024];
  char* gdd_file = new char[1024];
  if (do_vert) {
    strcat(vert_file, template_file);
    strcat(vert_file, ".vert");
  }
  if (do_gdd) {
    strcat(gdd_file, template_file);
    strcat(gdd_file, ".gdd");
  }

  Pattern patt;
  read_in_pattern(patt, template_file, labeled, srcs_t, dsts_t, labels_t);

  double full_count = 0.0;  
  if (do_outerloop) {
    int num_threads = omp_get_max_threads();
    int iter = ceil( (double)iterations / (double)num_threads + 0.5);
    colorcount* graph_count = new colorcount[num_threads];
    for (int tid = 0; tid < num_threads; ++tid) {
      graph_count[tid].init(g, labels_g, labeled, calc_auto, do_gdd, do_vert, verbose);
    }
    double** vert_counts;
    if (do_gdd || do_vert)
      vert_counts = new double*[num_threads];
    #pragma omp parallel reduction(+:full_count)
    {
    int tid = omp_get_thread_num();
    full_count += graph_count[tid].do_full_count(&patt, labels_t, iter);
    if (do_gdd || do_vert)
      vert_counts[tid] = graph_count[tid].get_vert_counts();
    }
    full_count /= (double)num_threads;
    if (do_gdd || do_vert) {
      output out(vert_counts, num_threads, g.num_vertices());
      if (do_gdd) {
        out.output_gdd(gdd_file);
        free(gdd_file);
      } 
      if (do_vert) {        
        out.output_verts(vert_file);
        free(vert_file);
      }
    }
  } else {
    colorcount graph_count;
    graph_count.init(g, labels_g, labeled, calc_auto, do_gdd, do_vert, verbose);
    full_count += graph_count.do_full_count(&patt, labels_t, iterations);

    if (do_gdd || do_vert) {
      double* vert_counts = graph_count.get_vert_counts();
      output out(vert_counts, g.num_vertices());
      if (do_gdd) {
        out.output_gdd(gdd_file);
        free(gdd_file);
      }
      if (do_vert) {
        out.output_verts(vert_file);
        free(vert_file);
      }
    }
  }
  printf("Count:\n\t%e\n", full_count);
  delete [] srcs_g;
  delete [] dsts_g;
  delete [] labels_g;
  delete [] srcs_t;
  delete [] dsts_t;
  delete [] labels_t;
}

