# GraphGPUMiner #

This repository provides the Graph Pattern Mining (GPM) Framework on GPU.

## Getting Started ##

The document is organized as follows:

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [Supported graph formats](#supported-graph-formats)
* [Code Documentation](#code-documentation)
* [Notes](#notes)
* [Reporting bugs and contributing](#reporting-bugs-and-contributing)
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

To control the number of threads, set the following environment variable:

    $ export OMP_NUM_THREADS=[ number of cores in system ]


### Supported graph formats ###

The graph loading infrastructure understands the following formats:

+ `graph.meta.txt` text file specifying the number of vertices, edges and maximum degree

+ `graph.vertex.bin` binary file containing the row pointers

+ `graph.edge.bin` binary file containing the column indices

+ `graph.vlabel.bin` binary file containing the vertax labels (only needed for labeled graphs)

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

### Notes ###

Three existing state-of-the-art frameworks:

Pangolin [1]: source code is in src/pangolin/.

SgMatch [2,3]: https://github.com/guowentian/SubgraphMatchGPU

Peregrine [4]: https://github.com/pdclab/peregrine

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali.
Pangolin: An Efficient and Flexible Graph Pattern Mining System on CPU and GPU. VLDB 2020

[2] Wentian Guo, Yuchen Li, Mo Sha, Bingsheng He, Xiaokui Xiao, Kian-Lee Tan.
GPU-Accelerated Subgraph Enumeration on Partitioned Graphs. SIGMOD 2020.

[3] Wentian Guo, Yuchen Li, Kian-Lee Tan. 
Exploiting Reuse for GPU Subgraph Enumeration. TKDE 2020.

[4] Kasra Jamshidi, Rakesh Mahadasa, Keval Vora.
Peregrine: A Pattern-Aware Graph Mining System. EuroSys 2020

### Reporting bugs and contributing ###

If you find any bugs please report them by using the repository (github **issues** panel).
We are also ready to engage in improving and extending the framework if you request new features.

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

## Developers ##

* `Xuhao Chen`, Postdoc, MIT, cxh@mit.edu

## License ##

> Copyright (c) 2017, MIT
> All rights reserved.
