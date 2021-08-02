k-Motif Counting (k-MC)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Author: Xuhao Chen <cxh@mit.edu>

This program counts the number of k-motifs in a given undirected graph.

This implementation reduces the search space by counting each k-motif only once.
This is done by establish partial orders among vertices in the input graph.
Meanwhile, each motif has a matching order which avoids graph isomorphism test.
An advanced optimization, called formula-based local counting, is used
to significantly prune the enumeration space. Only triangles are enumerated
in 3-motif counting. Similarly only 4-cliques and 4-cycles are enumerated
in 4-motif counting.

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

  - motif_omp_base : one thread per vertex using OpenMP
  - motif_omp_formula: formula-based local counting using OpenMP

RUN
--------------------------------------------------------------------------------

The following is an example command line.

`$ ../../bin/motif_omp_base ../../inputs/citeseer/graph 3`

`$ ../../bin/motif_omp_formula ../../inputs/citeseer/graph 4`

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
