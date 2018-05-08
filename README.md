# machine-learning-package
Implement useful machine learning method based on C++. The matrix operation relies on the eigen library and some methods uses opencv only for visualization.The machine method mainly include classify method such as SVM,logistic,decision tree,mlp etc, cluster method such as kmeans,gmm,dbscan,lvq etc and regression method such as linear regression.

## Installation
This software has only been tested on ubuntu 16.04(x64), eigen3, opencv3.4. To install this package your compiler need to support C++11. All the machine learning methods shared the same interface which is convenient to call or modify.
```
git clone https://github.com/TJCVRS/machine-learning-package.git
```
#### Required Dependencies
Eigen3, boost and opencv required. Install them with
```
sudo apt-get install libopencv-dev libeigen3-dev libboost-all-dev
```

## Build
```
cd ROOT_FOLDER
mkdir build
cd build
cmake ..
make -j
```
This will compile a binary file. You can modified the main_test.cpp and the CMakeLists.txt file to compile different machine learning method binary file.

## Usage
#### Gaussian Mixture Model for clustering
```
./bin_root/gmmCluster ../data/kmeans.txt ../data/kmeans.txt ../kmeans.txt
```
`Origin data distribution is as follows`

![gmm_cluster_ori](https://github.com/TJCVRS/machine-learning-package/blob/master/data/images/gmm_cluster_ori.jpg)

`Clustering visualization result is as follows`

![gmm_cluster_visualization](https://github.com/TJCVRS/machine-learning-package/blob/master/data/images/gmm_cluster.gif)

#### DBScan Model for clustering
```
./bin_root/dbscanCluster ../data/kmeans.txt ../data/kmeans.txt ../kmeans.txt
```
`Origin data distribution is as follows`

![dbscan_cluster_ori](https://github.com/TJCVRS/machine-learning-package/blob/master/data/images/dnscan_ori.png)

`Clustering visualization result is as follows`

![dbscan_cluster_visualization](https://github.com/TJCVRS/machine-learning-package/blob/master/data/images/dbscan_result.png)


