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

OUTPUT
--------------------------------------------------------------------------------

|                  |      mico     | citeseer |     patent    |     youtube     |  soc-Livejournal  |      com-orkut     |     twitter20     |      friendster     |       uk2007       |      twitter40      |      gsh-2015      |
|------------------|--------------:|---------:|--------------:|----------------:|------------------:|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
| 4-paths          | 4,070,868,075 |  111,153 | 5,764,763,466 |  55,176,204,040 | 1,147,811,961,320 | 18,573,723,211,463 |                   | 364,700,730,542,912 |                    |                     |                    |
| 3-stars          | 2,307,847,995 |  222,630 | 5,148,841,859 | 201,577,267,737 | 6,619,009,156,172 | 97,824,018,291,804 |                   | 247,358,335,700,296 |                    |                     |                    |
| 4-cycles         |    33,929,353 |    3,094 |   227,197,040 |     366,107,225 |     4,966,580,492 |     70,100,119,560 |                   |     307,502,615,265 |                    |                     |                    |
| tailed-triangles | 3,591,944,265 |   22,900 |   497,680,804 |   8,209,274,276 |   124,769,176,079 |  1,510,018,661,295 |                   |   5,787,076,338,289 |                    |                     |                    |
| diamonds         |   437,985,111 |    2,200 |    55,988,120 |     746,615,826 |    16,753,396,228 |     47,767,212,604 |                   |     131,410,239,292 |                    |                     |                    |
| 4-cliques        |   514,864,225 |      255 |     3,310,556 |     176,614,367 |     9,933,532,019 |      3,221,946,137 |                   |       8,963,503,263 |                    |                     |                    |
|                  |               |          |               |                 |                   |                    |                   |                     |                    |                     |                    |
| wedges           |    53,546,459 |   23,380 |   267,600,153 |   1,867,293,654 |     6,412,312,961 |     43,742,714,028 | 1,780,251,390,046 |     708,133,792,538 | 25,162,884,716,555 | 123,331,114,814,249 | 40,085,646,959,056 |
| triangles        |    12,534,960 |    1,166 |     6,913,764 |     103,017,122 |       285,730,264 |        627,584,181 |    17,295,646,010 |       4,173,724,142 |    286,701,284,103 |      34,824,916,864 |    910,140,734,636 |

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
