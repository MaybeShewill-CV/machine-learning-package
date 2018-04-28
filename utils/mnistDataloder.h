/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: mnistDataloder.h
* Date: 18-4-28 上午11:06
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_MNISTDATALODER_H
#define MACHINE_LEARNING_PACKAGE_MNISTDATALODER_H


#include <vector>
#include <eigen3/Eigen/Dense>

typedef std::pair<Eigen::VectorXd, double> mnist_sample;

class mnist_dataloder {
public:
    mnist_dataloder() = default;
    ~mnist_dataloder() = default;

    std::vector<mnist_sample> load_mnist(const std::string &mnist_file_path,
                                         const std::string &mnist_label_path);

};


#endif //MACHINE_LEARNING_PACKAGE_MNISTDATALODER_H
