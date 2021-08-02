Subgraph Listing (SL)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Author: Xuhao Chen <cxh@mit.edu>

This program counts the number of embeddings isomorphic to a given pattern in a 
given undirected graph.

This implementation reduces the search space by counting each embedding only once.
This is done by establishing partial ordering among vertices in the input graph.
To find embeddings, we start from each vertex v, and iteratively add one more 
vertex from its neighborhood. We use a connectivity map to record the neighborhood
connectivity. Meanwhile, a matching order specific to the pattern is used to avoid
graph isomorphism test.

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

  - sgl_omp_base : one thread per vertex using OpenMP

RUN
--------------------------------------------------------------------------------

The following is an example command line:

`$ ../../bin/sgl_omp_base ../../inputs/citeseer/graph diamond`

OUTPUT
--------------------------------------------------------------------------------

|                  |     4-cycles    |        diamonds       | tailed-triangles |        house       | pentagon           |
|------------------|----------------:|----------------------:|-----------------:|-------------------:|-------------------:|
| citeseer         |           6,059 |                 3,730 |                  |             55,359 |             28,394 |
| mico             |   2,016,507,139 |         3,527,170,461 |                  |  1,655,449,692,098 |    394,942,854,039 |
| patent_citations |     293,116,828 |            75,851,456 |                  |      6,586,768,851 |      3,254,769,712 |
| cit-Patents      |     341,906,226 |            83,785,566 |                  |      7,375,094,981 |      3,663,584,163 |
| youtube          |   1,642,566,152 |         1,806,302,028 |                  |     71,503,929,498 |     24,702,570,492 |
| livej            |  51,520,572,777 |        76,354,588,342 |                  | 53,552,979,463,652 | 13,892,452,066,046 |
| com-Orkut        | 127,533,170,575 |        67,098,889,426 |                  |                    |                    |
| Friendster       | 465,803,364,346 |       185,191,258,870 |                  |                    |                    |
| Twitter20        |                 |    41,166,070,788,458 |                  |                    |                    |
| Twitter40        |                 |   176,266,103,582,254 |                  |                    |                    |
| uk2007           |                 | 1,829,065,851,491,100 |                  |                    |                    |

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
