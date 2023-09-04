Triangle Counting (TC)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Author: Xuhao Chen <cxh@mit.edu>

This program counts the number of triangles in a given undirected graph.

This implementation reduces the search space by counting each triangle only
once. A naive implementation will count the same triangle six times because
each of the three vertices (u, v, w) will count it in both ways. To count
a triangle only once, this implementation only counts a triangle if u > v > w.
To setup this total ordering among vertices, the input undirected graph is 
converted into a directed acyclic graph (DAG). This technqiue is well known
as orientation.

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

  - tc_omp_base : one thread per vertex using OpenMP

RUN
--------------------------------------------------------------------------------

The following are example command lines:

`$ ../../bin/tc_omp_base ../../inputs/citeseer/graph`

OUTPUT
--------------------------------------------------------------------------------

|                  |    # nodes  |#edges(directed)| #edges(undirected)|   # triangles   |
|------------------|------------:|---------------:|------------------:|----------------:|
| citeseer         |        3,312|          4,536 |             9,072 |           1,166 |
| mico             |     100,000 |      1,080,156 |         2,160,312 |      12,534,960 |
| patent_citations |   2,745,761 |     13,965,409 |        27,930,818 |       6,913,764 |
| cit-Patents      |   3,774,768 |     16,518,947 |        33,037,894 |       7,515,023 |
| youtube          |   7,066,392 |     57,095,242 |       114,190,484 |     103,017,122 |
| livej            |   4,847,571 |     42,851,237 |        85,702,474 |     285,730,264 |
| com-Orkut        |   3,072,441 |    117,185,083 |       234,370,166 |     627,584,181 |
| twitter20        |  21,297,772 |    265,025,545 |       530,051,090 |  17,295,646,010 |
| twitter40        |  41,652,230 |  1,202,513,046 |     2,405,026,092 |  34,824,916,864 |
| friendster       |  65,608,366 |  1,806,067,135 |     3,612,134,270 |   4,173,724,142 |
| uk2007           | 105,896,435 |  3,301,876,564 |     6,603,753,128 | 286,701,284,103 |
| gsh-2015         | 988,490,691 | 25,690,705,118 |    51,381,410,236 | 910,140,734,636 |

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
