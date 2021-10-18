# GraphMinerBench #

GraphMinerBench is a C++ implemented Benchmark Suite for Graph Pattern Mining (GPM), based on the implementations of state-of-the-art GPM Frameworks including Pangolin [1], Sandslash [2] and FlexMiner [3]. GraphMinerBench supports both multicore CPU and GPU, and is parallelized using OpenMP and CUDA respectively. 

Unlike those GPM frameworks, GraphMinerBench is inherently designed for benchmarking hardware architecture design.
It includes various GPM workloads (e.g., TC, k-CL, SgL, k-MC, FSM) as well as representative graph datasets (e.g., Mico, Patents, Youtube, LiveJournal, Orkut, Twitter, Friendster). 

Please contat [the author](http://people.csail.mit.edu/xchen/) for accessing the datasets.

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali.
[Pangolin: An Efficient and Flexible Graph Pattern Mining System on CPU and GPU](http://www.vldb.org/pvldb/vol13/p1190-chen.pdf). VLDB 2020

[2] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Loc Hoang, Keshav Pingali.
[Sandslash: A Two-Level Framework for Efficient Graph Pattern Mining](http://people.csail.mit.edu/xchen/docs/ics-2021.pdf), ICS 2021

[3] Xuhao Chen, Tianhao Huang, Shuotao Xu, Thomas Bourgeat, Chanwoo Chung, Arvind.
[FlexMiner: A Pattern-Aware Accelerator for Graph Pattern Mining](http://people.csail.mit.edu/xchen/docs/isca-2021.pdf), ISCA 2021

## Getting Started ##

The document is organized as follows:

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [Supported graph formats](#supported-graph-formats)
* [Code Documentation](#code-documentation)
* [Reporting bugs and contributing](#reporting-bugs-and-contributing)
* [Notes](#notes)
* [Publications](#publications)
* [Developers](#developers)
* [License](#license)

### Requirements ###

* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) 11.1.1 or greater.
* GCC 8.3.1.
* CUB. if CUDA version < 11.0, enable CUB in the Makefile.

Note: the latest official CUB requires CUDA 11+. For CUDA version < 11.0, use CUB v1.8.0.

### Quick Start ###

Setup CUB library:

    $ git submodule update --init --recursive

Go to each sub-directory, e.g. src/triangle, and then

    $ cd src/triangle; make

Find out commandline format by running executable without argument:

    $ cd ../../bin
    $ ./tc_omp_base

Run triangle counting on an undirected graph:

    $ ./tc_omp_base ../inputs/citeseer/graph

You can find the corret output in the README of each benchmark [see here for triangle](https://github.com/chenxuhao/GraphMiner/blob/master/src/triangle/README.md).

To control the number of threads, set the following environment variable:

    $ export OMP_NUM_THREADS=[ number of cores in system ]


### Supported graph formats ###

The graph loading infrastructure understands the following formats:

+ `graph.meta.txt` text file specifying the number of vertices, edges and maximum degree

+ `graph.vertex.bin` binary file containing the row pointers

+ `graph.edge.bin` binary file containing the column indices

+ `graph.vlabel.bin` binary file containing the vertax labels (only needed for labeled graphs)

An example graph is in inputs/citeseer

Other graph input formats to be supported:

* Market (.mtx), [The University of Florida Sparse Matrix Collection](http://www.cise.ufl.edu/research/sparse/matrices/)
* Metis (.graph), [10th DIMACS Implementation Challenge](http://www.cc.gatech.edu/dimacs10/)
* SNAP (.txt), [Stanford Network Analysis Project](http://snap.stanford.edu/)
* Dimacs9th (.gr), [9th DIMACS Implementation Challenge](http://www.dis.uniroma1.it/challenge9/)
* The Koblenz Network Collection (out.< name >), [The Koblenz Network Collection](http://konect.uni-koblenz.de/)
* Network Data Repository (.edges), [Network Data Repository](http://networkrepository.com/index.php)
* Real-World Input Graphs (Misc), [Real-World Input Graphs](http://gap.cs.berkeley.edu/datasets.html)

### Code Documentation ###

The code documentation is located in the `docs` directory (*doxygen* html format).

### Reporting bugs and contributing ###

If you find any bugs please report them by using the repository (github **issues** panel).
We are also ready to engage in improving and extending the framework if you request new features.

## Notes ##

Existing state-of-the-art frameworks:

Pangolin [1]: source code is in src/pangolin/

SgMatch [2,3]: https://github.com/guowentian/SubgraphMatchGPU

Peregrine [4]: https://github.com/pdclab/peregrine

Sandslash [5]: source code is in src/\*/cpu_kernels/\*_cmap.h

FlexMiner [6]: the CPU baseline code is in \*/cpu_kernels/\*_base.h

DistTC [7]: source code is in src/triangle/

DeepGalois [8]: https://github.mit.edu/csg/DeepGraphBench

GraphPi [9]: https://github.com/thu-pacman/GraphPi

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali.
Pangolin: An Efficient and Flexible Graph Pattern Mining System on CPU and GPU. VLDB 2020

[2] Wentian Guo, Yuchen Li, Mo Sha, Bingsheng He, Xiaokui Xiao, Kian-Lee Tan.
GPU-Accelerated Subgraph Enumeration on Partitioned Graphs. SIGMOD 2020.

[3] Wentian Guo, Yuchen Li, Kian-Lee Tan. 
Exploiting Reuse for GPU Subgraph Enumeration. TKDE 2020.

[4] Kasra Jamshidi, Rakesh Mahadasa, Keval Vora.
Peregrine: A Pattern-Aware Graph Mining System. EuroSys 2020

[5] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Loc Hoang, Keshav Pingali.
Sandslash: A Two-Level Framework for Efficient Graph Pattern Mining, ICS 2021

[6] Xuhao Chen, Tianhao Huang, Shuotao Xu, Thomas Bourgeat, Chanwoo Chung, Arvind.
FlexMiner: A Pattern-Aware Accelerator for Graph Pattern Mining, ISCA 2021

[7] Loc Hoang, Vishwesh Jatala, Xuhao Chen, Udit Agarwal, Roshan Dathathri, Grubinder Gill, Keshav Pingali.
DistTC: High Performance Distributed Triangle Counting, HPEC 2019

[8] Loc Hoang, Xuhao Chen, Hochan Lee, Roshan Dathathri, Gurbinder Gill, Keshav Pingali.
Efficient Distribution for Deep Learning on Large Graphs, GNNSys 2021

[9] Tianhui Shi, Mingshu Zhai, Yi Xu, Jidong Zhai. 
GraphPi: high performance graph pattern matching through effective redundancy elimination. SC 2020

## Publications ##

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
@inproceedings{DistTC,
  title={DistTC: High performance distributed triangle counting},
  author={Hoang, Loc and Jatala, Vishwesh and Chen, Xuhao and Agarwal, Udit and Dathathri, Roshan and Gill, Gurbinder and Pingali, Keshav},
  booktitle={2019 IEEE High Performance Extreme Computing Conference (HPEC)},
  pages={1--7},
  year={2019},
  organization={IEEE}
}
```

```
@inproceedings{Sandslash,
  title={Sandslash: a two-level framework for efficient graph pattern mining},
  author={Chen, Xuhao and Dathathri, Roshan and Gill, Gurbinder and Hoang, Loc and Pingali, Keshav},
  booktitle={Proceedings of the ACM International Conference on Supercomputing},
  pages={378--391},
  year={2021}
}
```

```
@inproceedings{hoang2019disttc,
  title={DistTC: High performance distributed triangle counting},
  author={Hoang, Loc and Jatala, Vishwesh and Chen, Xuhao and Agarwal, Udit and Dathathri, Roshan and Gill, Gurbinder and Pingali, Keshav},
  booktitle={2019 IEEE High Performance Extreme Computing Conference (HPEC)},
  pages={1--7},
  year={2019},
  organization={IEEE}
}
```

```
@inproceedings{DeepGalois,
  title={Efficient Distribution for Deep Learning on Large Graphs},
  author={Hoang, Loc and Chen, Xuhao and Lee, Hochan and Dathathri, Roshan and Gill, Gurbinder and Pingali, Keshav},
  booktitle={Workshop on Graph Neural Networks and Systems},
  volume={1050},
  pages={1-9},
  year={2021}
}
```

## Developers ##

* [Xuhao Chen](http://people.csail.mit.edu/xchen/), Research Scientist, MIT, cxh@mit.edu
* [Tianhao Huang](https://nicsefc.ee.tsinghua.edu.cn/people/tianhao-h/), PhD student, MIT

## License ##

> Copyright (c) 2021, MIT
> All rights reserved.
