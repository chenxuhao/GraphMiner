## Subgraph Counting (SC)

### DESCRIPTION 

Author: Xuhao Chen <cxh@mit.edu>

This program counts the number of edge-induced subgraphs isomorphic to a given pattern in a given undirected graph.

This implementation reduces the search space by using the matching order and pattern decomposition.

### INPUT

The input graph is preprocessed internally to meet these requirements:

  - to be undirected

  - no self-loops

  - no duplicate edges (or else will be counted as multiple triangles)

  - neighborhoods are sorted by vertex identifiers

### BUILD

1. Run make at this directory

2. Or run make at the top-level directory

  - count_omp_base : one thread per vertex using OpenMP

### RUN

The following is an example command line:

`$ ../../bin/count_omp_base ../../inputs/citeseer/graph diamond`

### OUTPUT

|                  |     4-cycles        |        diamonds       |     hourglass    |        house       | pentagon           |
|------------------|--------------------:|----------------------:|-----------------:|-------------------:|-------------------:|
| citeseer         |           6,059     |                 3,730 |     16,034       |             55,359 |             28,394 |
| mico             |   2,016,507,139     |         3,527,170,461 |  519,582,993,122 |  1,655,449,692,098 |    394,942,854,039 |
| patent_citations |     293,116,828     |            75,851,456 |   1,210,636,555  |      6,586,768,851 |      3,254,769,712 |
| cit-Patents      |     341,906,226     |            83,785,566 |   1,315,892,087  |      7,375,094,981 |      3,663,584,163 |
| youtube          |   1,642,566,152     |         1,806,302,028 |  27,174,070,930  |     71,503,929,498 |     24,702,570,492 |
| livej            |  51,520,572,777     |        76,354,588,342 |15,019,714,040,315| 53,552,979,463,652 | 13,892,452,066,046 |
| com-Orkut        | 127,533,170,575     |        67,098,889,426 |28,809,758,775,904| 49,691,624,220,872 |                    |
| Friendster       | 465,803,364,346     |       185,191,258,870 |25,851,776,646,531| 54,693,897,472,726 |                    |
| Twitter20        |                     |    41,166,070,788,458 |                  |                    |                    |
| Twitter40        |                     |   176,266,103,582,254 |351,530,246,683,753,988|                    |                    |
| uk2007           |1,656,250,936,574,295| 1,829,065,851,491,105 |2,416,297,962,006,987,028|                    |                    |

### CITATION

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

```
@INPROCEEDINGS{FlexMiner,
  author={Chen, Xuhao and Huang, Tianhao and Xu, Shuotao and Bourgeat, Thomas and Chung, Chanwoo and Arvind},
  booktitle={2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)}, 
  title={FlexMiner: A Pattern-Aware Accelerator for Graph Pattern Mining}, 
  year={2021},
  volume={},
  number={},
  pages={581-594},
  doi={10.1109/ISCA52012.2021.00052}
}
```

```
@inproceedings {G2Miner,
author = {Xuhao Chen and Arvind},
title = {Efficient and Scalable Graph Pattern Mining on {GPUs}},
booktitle = {16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)},
year = {2022},
isbn = {978-1-939133-28-1},
address = {Carlsbad, CA},
pages = {857--877},
url = {https://www.usenix.org/conference/osdi22/presentation/chen},
publisher = {USENIX Association},
month = jul,
}
```
