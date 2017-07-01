# The k-means-u* clustering algorithm

![GitHub Logo](notebooks/img/example.png)

This repository contains example python code for the k-means-u and k-mean-u* algorithms as proposed in https://arxiv.org/abs/1706.09059. 

## Quick Start
* clone repository: `git clone https://github.com/gittar/k-means-u-star`
* cd main directory: `cd k-means-u-star`
* install miniconda or anaconda: https://conda.io/docs/install/quick.html
* create kmus environment: `conda env create -f envsimple.yml`
* activate environment: `source activate kmus` (on windows: `activate kmus`)
* start one of the jupyter notebooks: `jupyter notebook algo-pure.ipynb`
* continue in the browser window which opens (jupyter manual: http://jupyter-notebook.readthedocs.io/en/latest/)

## jupyter notebooks:
* algo-pure.ipynb <br>
  (a bare-bones implementation meant for easy understanding of the algorithms)
* simu-detail.ipynb <br>
  (detailed simulations and graphics to illustrate the way the algrithms work, uses kmeansu.py)
* simu-bulk.ipynb <br>
  (systematic simulations with various data sets to compare k-means-++, k-means-u and k-means-u*, uses kmeansu.py)
* dataset_class.ipynb<br>
  (examples for using the data generator)
  
## python files:
* kmeansu.py <br>
  main implementation of k-means-u and k-means-u*, makes heavy use of 
  http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for efficient implementations of k-means and k-means++, gathers certain statistics while training to enable systematic evaluation, code therefore a bit larger
* bfdataset.py <br>
  (contains a class "dataset" to generate test data sets and also an own implementation of k-means++ which allows to get  the codebook after initialization but before the run of k-means)
* bfutil.py <br>
  (various utility functions for plotting etc.)
