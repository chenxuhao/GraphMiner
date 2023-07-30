// Copyright (c) 2013, The Pennsylvania State University.
// All rights reserved.
// 
// See COPYING for license.

// Utility functions used by other classes

using namespace std;


double timer() {

    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);

}

int* dynamic_to_static(vector<int>& arr)
{
  int* new_array = new int[arr.size()];
  
  for (size_t i = 0; i < arr.size(); ++i)
    new_array[i] = arr[i];
  
  return new_array;
}

bool contains(vector<int>& arr, int item)
{
  for (size_t i = 0; i < arr.size(); ++i)
    if (arr[i] == item)
      return true;
  
  return false;
}


bool contains(int* arr, int length, int item)
{
  for (int i = 0; i < length; ++i)
    if (arr[i] == item)
      return true;
  
  return false;
}


void extract_uniques(vector<int>& source, vector<int>& dest)
{  
  for (size_t i = 0; i < source.size(); ++i)
    if (!contains(dest, source[i]))
      dest.push_back(source[i]);  
}


void sort(vector<int>& arr)
{
  bool swapped;
  
  do
  {
    swapped = false;
    for (size_t i = 1; i < arr.size(); ++i)
    {
      if (arr[i] < arr[i-1])
      {
        int temp = arr[i];
        arr[i] = arr[i-1];
        arr[i-1] = temp;
        swapped = true;
      }
    }
  } while (swapped);
}

void sort(int* arr, int size)
{
  bool swapped;
  
  do
  {
    swapped = false;
    for (int i = 1; i < size; ++i)
    {
      if (arr[i] < arr[i-1])
      {
        int temp = arr[i];
        arr[i] = arr[i-1];
        arr[i-1] = temp;
        swapped = true;
      }
    }
  } while (swapped);
}


void sort(double* arr, int size)
{
  bool swapped;
  
  do
  {
    swapped = false;
    for (int i = 1; i < size; ++i)
    {
      if (arr[i] < arr[i-1])
      {
        double temp = arr[i];
        arr[i] = arr[i-1];
        arr[i-1] = temp;
        swapped = true;
      }
    }
  } while (swapped);
}


int get_max(vector<int> arr)
{
  int maximum = 0; 
  int size = arr.size();

  for (int i = 0; i < size; i++)
  {
    if (maximum < arr[i])
    {
      maximum = arr[i];
    }
  }
  
  return maximum;
}


int get_max(vector<int> arr1, vector<int> arr2)
{
  int maximum = 0; 
  int size = arr1.size();

  for (int i = 0; i < size; i++)
  {
    if (maximum < arr1[i])
    {
      maximum = arr1[i];
    }
    if (maximum < arr2[i])
    {
      maximum = arr2[i];
    }
  }
  
  return maximum;
}

int get_max(int* arr, int length)
{
  int max = 0;
  for (int i = 0; i < length; ++i)
    if (arr[i] > max)
      max = arr[i];
      
  return max;
}

int* init_permutation(int num_verts)
{
  int* perm = new int[num_verts];
  
  for (int i = 0; i < num_verts; i++)
  {
    perm[i] = i + 1;
  }
  
  return perm;
}

/*
bool is_permutation(int* current_perm)
{
  for (int i = 0; i < sizeof_array(current_perm); i++)
    for (int j = i + 1; j < sizeof_array(current_perm); j++)
      if (current_perm[i] == current_perm[j])
        return false;
        
  return true;
}
*/

void next_set(int* current_set, int length, int num_colors)
{
  for (int i = length - 1; i >= 0; --i)
  {
    if (current_set[i] < num_colors - (length - i - 1))
    {
      current_set[i] = current_set[i] + 1;
      for (int j = i + 1; j < length; ++j)
      {
        current_set[j] = current_set[j-1] + 1;
      }
      break;
    }
  }
}

int factorial(int x)
{
  if (x <= 0)
  return 1;
  else
  return (x == 1 ? x : x * factorial(x - 1));
}


int choose(int n, int k)
{
  int num_c;

  if (n < k)
    num_c = 0;
  else
    num_c = (int) factorial(n) / (factorial(k) * factorial(n - k));

  
  return num_c;
}
/*
template <typename int>
void generate_all_permutations(vector<int> mapping, vector<int> rest, int** all_perms, int* counter)
{
  if (rest.size() == 0) 
  {
        for (int i = 0; i < mapping.size(); ++i)
    {
      //printf("%lu ", mapping[i]);
      all_perms[*counter][i] = mapping[i];
    }
    //printf("Done\n");
    *counter = *counter + 1;
    //printf("%lu\n", *counter);
    } 
  else   
  {
    for (int i = 0; i < rest.size(); ++i)
    {
      mapping.push_back(rest[i]);
      
      vector<int> new_rest;
      for (int j = 0; j < rest.size(); ++j)
      {
        if (i != j)
          new_rest.push_back(rest[j]);     
      }
      
      generate_all_permutations(mapping, new_rest, all_perms, counter);
            new_rest.clear();
      
      mapping.pop_back();      
        }
    }
}
*/


int test_automorphism(Pattern& t, vector<int>& mapping)
{  
  for (size_t v = 0; v < mapping.size(); ++v)
  {
    //printf("%lu\n", v);
    //printf("%lu\n", mapping[v]);
    if (t.out_degree(v) != t.out_degree(mapping[v]))
      return 0;
    else
    {
      int* adjs = t.adjacent_vertices(v);
      int* adjs_map = t.adjacent_vertices(mapping[v]);        
      int end = t.out_degree(v);
      
      bool* match = new bool[end];
      for (int i = 0; i < end; ++i)
      {
        match[i] = false;
        int u = adjs[i];
        for(int j = 0; j < end; ++j)
        {
          int u_map = adjs_map[j];
          //printf("%lu %lu %lu %lu\n", v, u, mapping[v], mapping[u_map]);
          if (u == mapping[u_map])
            match[i] = true;
        }
      }
          
      for (int i = 0; i < end; ++i)
        if (!match[i])
          return 0;
      /*
        vertex_descriptor u = adjs[i];
        vertex_descriptor u_map = adjs_map[i];
        
        printf("%lu %lu %lu %lu\n", v, u, u_map, mapping[u_map]);
        
        if (u != mapping[u_map])
          return 0;
      */
      
    }
  }

  //printf("found auto: ");
  //for (int v = 0; v < mapping.size(); ++v)
  //  printf("%lu ", mapping[v]);
  //printf("\n");
  
  return 1;
}

int count_all_automorphisms(Pattern& t, vector<int>& mapping, vector<int>& rest)
{
  int count = 0;
  
  if (rest.size() == 0) 
  {    
    //for (int i = 0; i < mapping.size(); ++i)
    //{
    //  printf("%lu ", mapping[i]);
    //}
    //printf("Done\n");
    count = test_automorphism(t, mapping);
    return count;
    } 
  else   
  {
    for (size_t i = 0;i < rest.size(); ++i)
    {
      mapping.push_back(rest[i]);
      
      vector<int> new_rest;
      for (size_t j = 0; j < rest.size(); ++j)
      {
        if (i != j)
          new_rest.push_back(rest[j]);     
      }
      
      count += count_all_automorphisms(t, mapping, new_rest);
            new_rest.clear();      
      mapping.pop_back();      
        }
    }
  
  return count;
}

int count_automorphisms(Pattern& t)
{  
  int count = 0;  
  int num_verts = t.num_vertices();
  
  //int num_trials = factorial(num_verts);
  vector<int> mapping;
  vector<int> rest;
  for (int i = 0; i < num_verts; ++i)
    rest.push_back(i);  
  
  count = count_all_automorphisms(t, mapping, rest);
  
  //int counter = 0;
  //int** all_perms = new int*[num_trials];  
  //for (int i = 0; i < num_trials; ++i)
  //  all_perms[i] = new int[num_verts];
  
  //generate_all_permutations(mapping, rest, all_perms, &counter);
  //printf("done gen\n");
  
  //int* permutation;
  
  /*
  for (int i = 0; i < num_trials; ++i)
  {
    permutation = next(permutation
    count += test_automorphism(t, all_perms[i], num_verts);
    //printf("count %lu\n", count);
  }  
  */
  return count;
}


bool test_isomorphism(Pattern& t, Pattern& h, vector<int>& mapping)
{  
  for (size_t v = 1; v < mapping.size(); ++v)
  {
    //printf("%lu\n", v);
    //printf("%lu\n", mapping[v]);
    if (h.out_degree(v) != t.out_degree(mapping[v]))
      return false;
    else
    {
      int* adjs = h.adjacent_vertices(v);
      int* adjs_map = t.adjacent_vertices(mapping[v]);        
      int end = h.out_degree(v);
      
      bool* match = new bool[end];
      for (int i = 0; i < end; ++i)
      {
        match[i] = false;
        int u = adjs[i];
        for(int j = 0; j < end; ++j)
        {
          int u_map = adjs_map[j];
          //printf("%lu %lu %lu %lu\n", v, u, mapping[v], u_map);
          if (mapping[u] == u_map)
            match[i] = true;
        }
      }
          
      //printf("Checking\n");
      for (int i = 0; i < end; ++i)
        if (!match[i])
          return false;
      /*
        vertex_descriptor u = adjs[i];
        vertex_descriptor u_map = adjs_map[i];
        
        printf("%lu %lu %lu %lu\n", v, u, u_map, mapping[u_map]);
        
        if (u != mapping[u_map])
          return 0;
      */
      
    }
  }

  //printf("found auto: ");
  //for (int v = 0; v < mapping.size(); ++v)
  //  printf("%lu ", mapping[v]);
  //printf("\n");
  
  return true;
}

bool check_all_isomorphisms(Pattern& t, Pattern& h, vector<int>& mapping, vector<int>& rest)
{
  bool count = false;
  
  if (rest.size() == 0) 
  {    
    //for (int i = 0; i < mapping.size(); ++i)
    //{
    //  printf("%lu ", mapping[i]);
    //}  
    //printf("\n");
    count = test_isomorphism(t, h, mapping);
    //printf("Done %d\n", count);
    return count;
    } 
  else   
  {
    for (size_t i = 0;i < rest.size(); ++i)
    {
      mapping.push_back(rest[i]);
      
      vector<int> new_rest;
      for (size_t j = 0; j < rest.size(); ++j)
      {
        if (i != j)
          new_rest.push_back(rest[j]);     
      }
      
      count = check_all_isomorphisms(t, h, mapping, new_rest);
            new_rest.clear();      
      mapping.pop_back();  

      if (count)
        return true;
        }
    }
  
  return count;
}

// check rooted isomorphisms for unlabeled templates
bool check_rooted_isomorphism(Pattern& t, Pattern& h)
{
  
  int num_verts_t = t.num_vertices();
  int num_verts_h = h.num_vertices();
  
  if (num_verts_t != num_verts_h)
    return false;
  else if (t.out_degree(0) != h.out_degree(0))
    return false;
  else if (num_verts_t < 5)
    return true;
  else 
  {
    vector<int> mapping;
    vector<int> rest;
    mapping.push_back(0);
    for (int i = 1; i < num_verts_t; ++i)
      rest.push_back(i);    
  
    return check_all_isomorphisms(t, h, mapping, rest);
  }
    
}


int get_color_index(int* colorset, int length)
{
  int count = 0;

  for (int i = 0; i < length; ++i)
  {
    int n = colorset[i] - 1;
    int k = i + 1;

    count += choose(n, k);
  }

  return count;
}  
  
  
int** init_choose_table(int num_colors)
{
  int** choose_table = new int*[num_colors + 1];

  for (int i = 0; i <= num_colors; ++i)
    choose_table[i] = new int[num_colors + 1];
    
  for (int i = 0; i <= num_colors; ++i)
    for (int j = 0; j <= num_colors; ++j)
      choose_table[i][j] = choose(i, j);
      
  return choose_table;
}
