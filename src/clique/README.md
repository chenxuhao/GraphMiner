k-Clique Listing (k-CL)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Author: Xuhao Chen <cxh@mit.edu>

This program counts the number of k-cliques in a given undirected graph.

This implementation reduces the search space by counting each k-clique only once.
This is done by establishing a total ordering among vertices in the input graph.
To setup this total ordering, the input undirected graph is converted into
a directed acyclic graph (DAG). This technqiue is well known as orientation.
To find k-cliques, we start from each vertex v, and iteratively add one more 
vertex from its neighborhood. We use a connectivity map to record the neighborhood
connectivity. Each time a vertex is added, the map is queried to make sure
the new vertex is connected with all vertices in the embedding.

INPUT
--------------------------------------------------------------------------------

The input graph is preprocessed internally to meet these requirements:

  - to be undirected

  - no self-loops

  - no duplicate edges (or else will be counted as multiple triangles)

  - neighborhoods are sorted by vertex identifiers

BUILD
--------------------------------------------------------------------------------

1. Run make at this directory

2. Or run make at the top-level directory

  - clique_omp_base : one thread per vertex using OpenMP

RUN
--------------------------------------------------------------------------------

The following is an example command line:

`$ ../../bin/clique_omp_base ../../inputs/citeseer/graph 4`

OUTPUT
--------------------------------------------------------------------------------

|                  |    m (q1)   |    nnz (q2)    | nnz (symmetrized) |  q3 (triangle)  |       4-clique      |       5-clique      |      6-clique      |     7-clique    |      8-clique     | 9-clique           |
|------------------|------------:|---------------:|------------------:|----------------:|--------------------:|--------------------:|-------------------:|----------------:|------------------:|-------------------:|
| mico             |     100,000 |      1,080,156 |         2,160,312 |      12,534,960 |         514,864,225 |      19,246,558,419 |    631,568,259,280 |                 |                   |                    |
| patent_citations |   2,745,761 |     13,965,409 |        27,930,818 |       6,913,764 |           3,310,556 |           2,976,152 |          3,132,860 |       1,870,484 |           515,317 |             61,358 |
| cit-Patents      |   3,774,768 |     16,518,947 |        33,037,894 |       7,515,023 |           3,501,071 |                     |                    |                 |                   |                    |
| youtube          |   7,066,392 |     57,095,242 |       114,190,484 |     103,017,122 |         176,614,367 |         295,551,667 |                    |                 |                   |                    |
| livej            |   4,847,571 |     42,851,237 |        85,702,474 |     285,730,264 |       9,933,532,019 |     467,429,836,174 | 20,703,476,954,640 |                 |                   |                    |
| com-Orkut        |   3,072,441 |    117,185,083 |       234,370,166 |     627,584,181 |       3,221,946,137 |      15,766,607,860 |     75,249,427,585 | 353,962,921,685 | 1,632,691,821,296 |  7,248,102,160,867 |
| twitter20        |  21,297,772 |    265,025,545 |       530,051,090 |  17,295,646,010 |   2,123,679,707,619 | 262,607,691,785,539 |                    |                 |                   |                    |
| twitter40        |  41,652,230 |  1,202,513,046 |     2,405,026,092 |  34,824,916,864 |   6,622,234,180,319 |                     |                    |                 |                   |                    |
| friendster       |  65,608,366 |  1,806,067,135 |     3,612,134,270 |   4,173,724,142 |       8,963,503,263 |      21,710,817,218 |     59,926,510,355 | 296,858,496,789 | 3,120,447,373,827 | 40,033,489,612,826 |
| uk2007           | 105,896,435 |  3,301,876,564 |     6,603,753,128 | 286,701,284,103 | 123,046,503,809,139 |                     |                    |                 |                   |                    |
| gsh-2015         | 988,490,691 | 25,690,705,118 |    51,381,410,236 | 910,140,734,636 | 205,010,080,145,349 |                     |                    |                 |                   |                    |

CITATION
--------------------------------------------------------------------------------

Please cite the following paper if you use this code:

```
@article{Pangolin,
	title={Pangolin: An Efficient and Flexible Graph Mining System on CPU and GPU},
	author={Xuhao Chen and Roshan Dathathri and Gurbinder Gill and Keshav Pingali},
	year={2020},
	journal = {Proc. VLDB Endow.},
	issue_date = {August 2020},
	volume = {13},
	number = {8},
	month = aug,
	year = {2020},
	numpages = {12},
	publisher = {VLDB Endowment},
}
```

```
@InProceedings{Sandslash,
	title={{Sandslash: A Two-Level Framework for Efficient Graph Pattern Mining}},
	author={Xuhao Chen and Roshan Dathathri and Gurbinder Gill and Loc Hoang and Keshav Pingali},
	year={2021},
	booktitle = {Proceedings of the 35th ACM International Conference on Supercomputing},
	series = {ICS '21},
	year = {2021},
	numpages = {14},
}
```
