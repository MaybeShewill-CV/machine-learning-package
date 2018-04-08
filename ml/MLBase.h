/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: MLBase.h
* Date: 18-4-3 下午2:51
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_MLBASE_H
#define MACHINE_LEARNING_PACKAGE_MLBASE_H

#include <eigen3/Eigen/Dense>

class MLBase {
public:
    MLBase() = default;
    virtual ~MLBase() = default;

    virtual void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) = 0;
    virtual void predict(const Eigen::MatrixXd& X, Eigen::MatrixXd& RET) = 0;

};


#endif //MACHINE_LEARNING_PACKAGE_MLBASE_H
