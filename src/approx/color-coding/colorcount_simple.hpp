#pragma once

#include "graph.h"
#include "pattern.hpp"

using namespace std;

class colorcount{
public:
    
  colorcount() {};  
  ~colorcount()
  {  }
    
  void init(Graph& full_graph, int* labels, bool label, 
            bool calc_auto, bool do_gdd, bool do_vert, bool verb)
  {
    g = &full_graph;
    num_verts_graph = g->num_vertices();
    labels_g = labels;
    labeled = label;    
    do_graphlet_freq = do_gdd;
    do_vert_output = do_vert;
    calculate_automorphisms = calc_auto;
    verbose = verb;

    if (do_graphlet_freq || do_vert_output)
    {
      final_vert_counts = new double[num_verts_graph];
      for (int i = 0; i < num_verts_graph; ++i)
        final_vert_counts[i] = 0.0;
    }
  }
  
  void init(Graph& full_graph, int* labels, bool label, 
            bool calc_auto, bool do_gdd, bool do_vert,
            int thread_id, bool verb)
  {
    g = &full_graph;
    num_verts_graph = g->num_vertices();
    labels_g = labels;
    labeled = label;
    do_graphlet_freq = do_gdd;
    do_vert_output = do_vert;
    calculate_automorphisms = calc_auto;
    verbose = verb;

    if (do_graphlet_freq || do_vert_output)
    {
      final_vert_counts = new double[num_verts_graph];
      for (int i = 0; i < num_verts_graph; ++i)
        final_vert_counts[i] = 0.0;
    }
  }

  double do_full_count(Pattern* sub_graph, int* labels, int N) {  
    num_iter = N;
    patt = sub_graph;
    labels_t = labels;          
    
    // create subtemplates and sort them in increasing order 
    part = partitioner(*patt, labeled, labels_t);
    part.sort_subtemplates();

    num_colors = patt->num_vertices();
    subtemplates = part.get_subtemplates();
    subtemplate_count = part.get_subtemplate_count();    

    create_tables();
    dt.init(subtemplates, subtemplate_count, g->num_vertices(), num_colors);


    // determine max out degree
    int max_out_degree = 0;
    for (int i = 0; i < num_verts_graph; i++) {
        int out_degree_i = g->out_degree(i);
        if (out_degree_i > max_out_degree) {
            max_out_degree = out_degree_i;
        }
    }
    max_degree = max_out_degree;

    // begin the counting    
    double count = 0.0;      
    for (int j = 0; j < N; j++) {
      count += template_count(); 
    }

    double final_count = count / (double) N;
    double prob_colorful = factorial(num_colors) / 
        ( factorial(num_colors - patt->num_vertices()) * pow(num_colors, patt->num_vertices()) );
    int num_auto = calculate_automorphisms ? count_automorphisms(*patt) : 1;    
    final_count = floor(final_count / (prob_colorful * (double)num_auto) + 0.5);

    if (do_graphlet_freq || do_vert_output)
    {
      for (int i = 0; i < num_verts_graph; ++i)
        final_vert_counts[i] = 
            (double)floor( final_vert_counts[i] / ((double)num_auto * 
            (double)N * prob_colorful) + 0.5);
    }
  
    delete_tables();
    part.clear_temparrays();  
    return final_count;      
  }

  double* get_vert_counts()
  {
    return final_vert_counts;
  }

private:
  // This does a single counting for a given templates
  // Return the full scaled count for the template on the whole graph
  double template_count()
  {  
    // do random coloring for full graph
    int num_verts = g->num_vertices();
    colors_g = new int[num_verts];
    
#pragma omp parallel 
{
    /* thread-local RNG initialization */
    r123::Philox4x32 rng;
    r123::Philox4x32::ctr_type rng_ctr = {{}};
    r123::Philox4x32::ukey_type rng_uk={{}};
    rng_uk[0] = 42; /* user-supplied key */
    r123::Philox4x32::key_type rng_key = rng_uk;

#pragma omp for
    for (int v = 0; v < num_verts; ++v)
    {
      rng_ctr[0] = v;
      rng_ctr[1] = cur_iter;      
      r123::Philox4x32::ctr_type r = rng(rng_ctr, rng_key);

      int color = r[0] % num_colors;
      colors_g[v] = color;
    }
}
    
    // start doing the counting, starting at bottom of partition tree
    //  then go through all of the subtemplates
    float full_count = 0.0;
    for (int s = subtemplate_count - 1; s >= 0 ; --s)
    {
      int num_verts_sub = num_verts_table[s];
      int a = part.get_active_index(s);
      int p = part.get_passive_index(s);   
      dt.init_sub(s, a, p);

      if (num_verts_sub == 1) 
        init_table_node(s);
      else
        full_count = colorful_count_simple(s);

      // remove table entries for children of this subtemplate   
      if (a != NULL_VAL)
        dt.clear_sub(a);
      if (p != NULL_VAL)
        dt.clear_sub(p);
    }
    delete [] colors_g; 

    return (double)full_count;
  }
  
  void init_table_node(int s)
  {
    int set_count_loop = 0;    

    if (!labeled)
    {
#pragma omp parallel for reduction(+:set_count_loop)
      for (int v = 0; v < num_verts_graph; ++v)
      {  
        int n = colors_g[v]; 
        dt.set(s, v, comb_num_indexes_set[s][n], 1.0);
        set_count_loop++;
      }
    }
    else
    {
      int* labels_sub = part.get_labels(s);  
      int label_s = labels_sub[0];
#pragma omp parallel for reduction(+:set_count_loop)
      for (int v = 0; v < num_verts_graph; ++v)
      {  
        int n = colors_g[v];
        int label_g = labels_g[v];
        if (label_g == label_s)
        {
          dt.set(s, v, comb_num_indexes_set[s][n], 1.0);
          set_count_loop++;
        }
      }
    }

    set_count = set_count_loop;
  }
  
  float colorful_count_simple(int s)
  {
    float cc = 0.0;   
    int active_index = part.get_active_index(s);
    int passive_index = part.get_passive_index(s);
    int num_verts_sub = num_verts_table[s]; 
    int num_verts_a = num_verts_table[active_index];  
    int num_combinations_sub = choose_table[num_colors][num_verts_sub]; 
    int num_combinations = choose_table[num_verts_sub][num_verts_a];     

#pragma omp parallel for schedule(static) reduction(+:cc) 
    for (int v = 0; v < num_verts_graph; ++v)
    { 
      int* adjs = g->adjacent_vertices(v);
      int end = g->out_degree(v);

      for (int n = 0; n < num_combinations_sub; ++n)
      {
        float color_count = 0.0;                
        int* comb_indexes_a = comb_num_indexes[0][s][n];
        int* comb_indexes_p = comb_num_indexes[1][s][n];

        int a = 0;         
        int p = num_combinations - 1;
        for (; a < num_combinations; ++a, --p) {
          for (int i = 0; i < end; ++i) {
            color_count += dt.get(active_index, v, comb_indexes_a[a]) * 
                dt.get(passive_index, adjs[i], comb_indexes_p[p]);
          }
        }
          
        cc += color_count;
        if (s != 0)
          dt.set(s, v, comb_num_indexes_set[s][n], color_count);
        else if (do_graphlet_freq || do_vert_output)
          final_vert_counts[v] += (double)color_count;          
      }
    }

    return cc;
  }
 
  // Creates all of the tables used, the important one is the combinatorial 
  //  number system indexes, as that is used for the gets and sets with the
  //  dynamic table 
  void create_tables()
  {
    choose_table = init_choose_table(num_colors);
    create_num_verts_table();
    create_all_index_sets();
    create_all_color_sets();
    create_comb_num_system_indexes();
    delete_all_color_sets();
    delete_all_index_sets();
  }  
  
  void delete_tables()
  {
    for (int i = 0; i <= num_colors; ++i)
      delete [] choose_table[i];    
    delete [] choose_table;

    delete_comb_num_system_indexes();
    delete [] num_verts_table;  
  }
  
  void create_num_verts_table()
  {
    num_verts_table = new int[subtemplate_count];    
    for (int s = 0; s < subtemplate_count; ++s)
      num_verts_table[s] = subtemplates[s].num_vertices();
  }
  
  void create_all_index_sets()
  {  
    index_sets = new int***[num_colors];
    
    for (int i = 0; i < (num_colors-1); ++i)
    {
      int num_vals = i + 2;
      index_sets[i] = new int**[(num_vals-1)];
      for (int j = 0; j < (num_vals-1); ++j)
      {  
        int set_size = j + 1;
        int num_combinations = choose(num_vals, set_size);
        index_sets[i][j] = new int*[num_combinations];
        
        int* set = init_permutation(set_size);
        
        for (int k = 0; k < num_combinations; ++k)
        {
          index_sets[i][j][k] = new int[set_size];
          for (int p = 0; p < set_size; ++p)
          {
            index_sets[i][j][k][p] = set[p] - 1;
          }
          next_set(set, set_size, num_vals);
        }
          
        delete [] set;
      }
    }
  }
  
  void delete_all_index_sets()
  {
    for (int i = 0; i < (num_colors-1); ++i)
    {
      int num_vals = i + 2;      
      for (int j = 0; j < (num_vals-1); ++j)
      {  
        int set_size = j + 1;
        int num_combinations = choose(num_vals, set_size);        
        for (int k = 0; k < num_combinations; ++k)
        {
          delete [] index_sets[i][j][k];
        }          
        delete [] index_sets[i][j];
      }      
      delete [] index_sets[i];
    }    
    delete [] index_sets;
  }  
  
  void create_all_color_sets()
  {
    color_sets = new int****[subtemplate_count];
    
    for (int s = 0; s < subtemplate_count; ++s)
    {
      int num_verts_sub = subtemplates[s].num_vertices();
      
      if (num_verts_sub > 1)
      {      
        int num_sets = choose(num_colors, num_verts_sub);
        color_sets[s] = new int***[num_sets];        

        int* colorset = init_permutation(num_verts_sub);
        for (int n = 0; n < num_sets; ++n)
        {
          int num_child_combs = num_verts_sub - 1;
          color_sets[s][n] = new int**[num_child_combs];
          
          for (int c = 0; c < num_child_combs; ++c)
          {
            int num_verts_1 = c+1;
            int num_verts_2 = num_verts_sub - num_verts_1;
            int** index_set_1 = index_sets[(num_verts_sub-2)][(num_verts_1-1)];
            int** index_set_2 = index_sets[(num_verts_sub-2)][(num_verts_2-1)];
        
            int num_child_sets = choose(num_verts_sub, (c + 1));
            color_sets[s][n][c] = new int*[num_child_sets];
            
            for (int i = 0; i < num_child_sets; ++i)
            {
              color_sets[s][n][c][i] = new int[num_verts_sub];
              
              for (int j = 0; j < num_verts_1; ++j)
                color_sets[s][n][c][i][j] = colorset[index_set_1[i][j]];
              for (int j = 0; j < num_verts_2; ++j)
                color_sets[s][n][c][i][j+num_verts_1] = colorset[index_set_2[i][j]];
            }
          }
          next_set(colorset, num_verts_sub, num_colors);
        }
        delete [] colorset;
      }
      
    }
  }
  
  void delete_all_color_sets()
  {
    for (int s = 0; s < subtemplate_count; ++s)
    {
      int num_verts_sub = subtemplates[s].num_vertices();      
      if (num_verts_sub > 1)
      {      
        int num_sets = choose(num_colors, num_verts_sub);        
        for (int n = 0; n < num_sets; ++n)
        {
          int num_child_combs = num_verts_sub - 1;
          for (int c = 0; c < num_child_combs; ++c)
          {
            int num_child_sets = choose(num_verts_sub, (c + 1));
            for (int i = 0; i < num_child_sets; ++i)
            {
              delete [] color_sets[s][n][c][i];              
            }            
            delete [] color_sets[s][n][c];            
          }
          delete [] color_sets[s][n];
        }          
        delete [] color_sets[s];
      }      
    }    
    delete [] color_sets;
  }
  
  void create_comb_num_system_indexes()
  {
    comb_num_indexes = new int***[2];
    comb_num_indexes[0] = new int**[subtemplate_count];
    comb_num_indexes[1] = new int**[subtemplate_count];    
    comb_num_indexes_set = new int*[subtemplate_count];
    
    for (int s = 0; s < subtemplate_count; ++s)
    {
      int num_verts_sub = subtemplates[s].num_vertices();      
      int num_combinations_s = choose(num_colors, num_verts_sub);
      
      if (num_verts_sub > 1)
      {  
        comb_num_indexes[0][s] = new int*[num_combinations_s];
        comb_num_indexes[1][s] = new int*[num_combinations_s];
      }
      comb_num_indexes_set[s] = new int[num_combinations_s];
      int* colorset_set = init_permutation(num_verts_sub);
      
      for (int n = 0; n < num_combinations_s; ++n)
      {      
        comb_num_indexes_set[s][n] = get_color_index(colorset_set, num_verts_sub);
      
        if (num_verts_sub > 1)
        {  
          int num_verts_a = part.get_num_verts_active(s);
          int num_verts_p = part.get_num_verts_passive(s);
      
          int* colors_a;        
          int* colors_p;
          int** colorsets = color_sets[s][n][num_verts_a - 1];
          
      
          int num_combinations_a = choose(num_verts_sub, num_verts_a);
          comb_num_indexes[0][s][n] = new int[num_combinations_a];        
          comb_num_indexes[1][s][n] = new int[num_combinations_a];        
          
          int p = num_combinations_a - 1;
          for (int a = 0; a < num_combinations_a; ++a, --p)
          {  
            colors_a = colorsets[a];          
            colors_p = colorsets[p] + num_verts_a;
            
            int color_index_a = get_color_index(colors_a, num_verts_a);
            int color_index_p = get_color_index(colors_p, num_verts_p);  

            comb_num_indexes[0][s][n][a] = color_index_a;
            comb_num_indexes[1][s][n][p] = color_index_p;
          }
        }        
        next_set(colorset_set, num_verts_sub, num_colors);
      }

       delete [] colorset_set;
    }
  }
  
  void delete_comb_num_system_indexes()
  {
    for (int s = 0; s < subtemplate_count; ++s)
    {
      int num_verts_sub = subtemplates[s].num_vertices();      
      int num_combinations_s = choose(num_colors, num_verts_sub);
      
      for (int n = 0; n < num_combinations_s; ++n)
      {  
        if (num_verts_sub > 1)
        {  
          delete [] comb_num_indexes[0][s][n];        
          delete [] comb_num_indexes[1][s][n];
        }
      }
      
      if (num_verts_sub > 1)
      {  
        delete [] comb_num_indexes[0][s];
        delete [] comb_num_indexes[1][s];
      }
      
      delete [] comb_num_indexes_set[s];
    }
    
    delete [] comb_num_indexes[0];
    delete [] comb_num_indexes[1];
    delete [] comb_num_indexes; 
    delete [] comb_num_indexes_set;
  }
  
  
  
  Graph* g;  // full graph
  Pattern* patt;  // template
  Pattern* subtemplates;
  int* labels_g;  
  int* labels_t;
  int* colors_g;
  bool labeled;
    
  int subtemplate_count;
  int num_colors;
  int num_iter;
  int cur_iter;
  
  dynamic_table_array dt;
  partitioner part;
  
  int** choose_table;
  int**** index_sets;
  int***** color_sets;
  int**** comb_num_indexes;
  int** comb_num_indexes_set;
  int* num_verts_table;  
  int num_verts_graph;
  int max_degree;
  
  double* final_vert_counts;
  bool do_graphlet_freq;
  bool do_vert_output;
  bool calculate_automorphisms;
  bool verbose;
  
  int set_count;
  int total_count;
  int read_count;
};


