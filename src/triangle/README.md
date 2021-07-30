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

Decompress the file and put it in the 'datasets' sub-directory:

    $ cd ../../datasets
    $ tar zxvf citeseer.tar.gz

The following are example command lines:

`$ ../../bin/tc_omp_base ../../datasets/citeseer/graph`

PERFORMANCE
--------------------------------------------------------------------------------

Please see details in the paper.

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

