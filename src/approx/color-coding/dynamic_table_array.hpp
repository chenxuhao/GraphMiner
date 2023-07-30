#pragma once

#include "pattern.hpp"

using namespace std;

class dynamic_table_array : public dynamic_table {
public:

  dynamic_table_array() {  
    table = NULL;
  }
  void init(Pattern* subs, int num_subtemplates, int num_vertices, int num_cols) {  
    subtemplates = subs;
    num_subs = num_subtemplates;
    num_verts = num_vertices;
    num_colors = num_cols;  
    init_choose_table();
    init_num_colorsets();
  
    table = (float***) malloc(num_subs * sizeof(float **));
    assert(table != NULL);
    is_sub_inited = (bool *)  malloc(num_subs * sizeof(bool));
    assert(is_sub_inited != NULL);

    for (int s = 0; s < num_subs; ++s) {
      is_sub_inited[s] = false;
    }  
    
    is_inited = true;
  }
  
  void init_sub(int subtemplate) {
    table[subtemplate] = (float**) malloc(num_verts * sizeof(float*));
    assert(table[subtemplate] != NULL);
    cur_table = table[subtemplate];
    cur_sub = subtemplate;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int v = 0; v < num_verts; v++) {
        cur_table[v] = NULL;
    }

    //int num_colorsets = 
    //    choose_table[num_colors][subtemplates[subtemplate].num_vertices()];
    is_sub_inited[subtemplate] = true;  
  }

  void init_sub(int subtemplate, int active_child, int passive_child)
  {
    if (active_child != NULL_VAL && passive_child != NULL_VAL)
    {
      cur_table_active = table[active_child];
      cur_table_passive = table[passive_child];
    }
    else
    {
      cur_table_active = NULL;
      cur_table_passive = NULL;
    }

    if (subtemplate != 0)
      init_sub(subtemplate);
  }
  
  void clear_sub(int subtemplate)
  {
    for (int v = 0; v < num_verts; ++v) {
      if (table[subtemplate][v])
        free(table[subtemplate][v]);
    }

    if (is_sub_inited[subtemplate]) {    
      free(table[subtemplate]);
    }

    is_sub_inited[subtemplate] = false;
  }
  
   
  void clear_table()
  {      
    for (int s = 0; s < num_subs; s++)
    {
      if (is_sub_inited[s]) 
      {
        for (int v = 0; v < num_verts; v++) {
          if (table[s][v])
            free(table[s][v]);
        }

        free(table[s]);
        is_sub_inited[s] = false;
      }
    }
  
    free(table);
    free(is_sub_inited);
  } 

  float get(int subtemplate, int vertex, int comb_num_index)
  {
    if (table[subtemplate][vertex]) {
      float retval = table[subtemplate][vertex][comb_num_index];
      return retval;
    } 
    else
      return 0.0;
  }   

  float get_active(int vertex, int comb_num_index)
  {
    if (cur_table_active[vertex]) {
      float retval = cur_table_active[vertex][comb_num_index];
      return retval;
    } 
    else
      return 0.0;
  }  

  float get_passive(int vertex, int comb_num_index)
  {
    if (cur_table_passive[vertex]) {
      float retval = cur_table_passive[vertex][comb_num_index];
      return retval;
    } 
    else
      return 0.0;
  } 

  float* get(int subtemplate, int vertex)
  {
    return table[subtemplate][vertex];
  }

  float* get_active(int vertex)
  {
    return cur_table_active[vertex];
  }

  float* get_passive(int vertex)
  {
    return cur_table_passive[vertex];
  }

  void set(int subtemplate, int vertex, int comb_num_index, float count)
  {
    if (table[subtemplate][vertex] == NULL)  
    {
      table[subtemplate][vertex] = 
                (float*)malloc(num_colorsets[subtemplate] * sizeof(float));
      assert(cur_table[vertex] != NULL);
      for (int c = 0; c < num_colorsets[subtemplate]; ++c) {      
        table[subtemplate][vertex][c] = 0.0;
      }
    }

    table[subtemplate][vertex][comb_num_index] = count;
  } 

  void set(int vertex, int comb_num_index, float count)
  {
    if (cur_table[vertex] == NULL)  
    {
      cur_table[vertex] = 
                (float*)malloc(num_colorsets[cur_sub] * sizeof(float));
      assert(cur_table[vertex] != NULL);
      for (int c = 0; c < num_colorsets[cur_sub]; ++c) {      
        cur_table[vertex][c] = 0.0;
      }
    }

    cur_table[vertex][comb_num_index] = count;
  } 
  
  bool is_init()
  {
    return is_inited;
  }
  
  bool is_sub_init(int subtemplate)
  {
    return is_sub_inited[subtemplate];
  }
  
  bool is_vertex_init_active(int vertex)
  {
    if (cur_table_active[vertex])
      return true;
    else
      return false;
  }

  bool is_vertex_init_passive(int vertex)
  {
    if (cur_table_passive[vertex])
      return true;
    else
      return false;
  } 

private:
  float*** table; 
  float** cur_table;
  float** cur_table_active;
  float** cur_table_passive;

  int cur_sub;
};
