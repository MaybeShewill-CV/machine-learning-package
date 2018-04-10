/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: logisticRegression.cpp
* Date: 18-4-10 下午2:39
************************************************/

#include "logisticRegression.h"

#include <glog/logging.h>

#define DEBUG

logisticRegression::logisticRegression(double learning_rate, int iter_times):
        lr(learning_rate), iter_times(iter_times){}

void logisticRegression::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 初始化权重
    auto feats_dims = static_cast<int>(X.cols());
    init_kernel(feats_dims);

    // bp更新权重
    LOG(INFO) << "开始训练:" << std::endl;
    for (auto i = 0; i < iter_times; ++i) {
        double loss = compute_loss(X, Y);
        if (std::abs(loss) < 0.0000001) {
            LOG(INFO) << "迭代训练完毕" << std::endl;
            return;
        }

        LOG(INFO) << "Iter: " << i << " Loss: " << loss << std::endl;

        forward_backward_update(X, Y);

#ifdef DEBUG
        Eigen::RowVectorXd gradient = compute_kernel_gradient(X, Y);
        LOG(INFO) << "参数更新权重为: " << gradient << std::endl;
#endif
    }
}

void logisticRegression::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::RowVectorXd foward_ret = forward(X);
    Eigen::MatrixXd ret_tmp(foward_ret.cols(), 1);
    for (auto i = 0; i < foward_ret.cols(); ++i) {
        ret_tmp(i, 0) = foward_ret(0, i);
    }
    RET = ret_tmp;
}

void logisticRegression::init_kernel(const int feats_dims) {
    kernel = Eigen::RowVectorXd::Random(feats_dims + 1);
}

Eigen::RowVectorXd logisticRegression::forward(Eigen::MatrixXd X) {
    assert(kernel.cols() == X.cols() + 1);

    Eigen::MatrixXd X_tmp(X.rows(), X.cols() + 1);
    for (auto i = 0 ; i < X.cols(); ++i) {
        X_tmp.col(i) = X.col(i);
    }
    X_tmp.col(X_tmp.cols() - 1) = Eigen::MatrixXd::Ones(X.rows(), 1);

    Eigen::RowVectorXd forward_ret(X_tmp.rows());
    forward_ret = X_tmp * kernel.transpose();

    forward_ret = sigmoid(forward_ret);

    return forward_ret;
}

double logisticRegression::compute_loss(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    assert(kernel.cols() == X.cols() + 1);

    Eigen::MatrixXd X_tmp(X.rows(), X.cols() + 1);
    for (auto i = 0 ; i < X.cols(); ++i) {
        X_tmp.col(i) = X.col(i);
    }
    X_tmp.col(X_tmp.cols() - 1) = Eigen::MatrixXd::Ones(X.rows(), 1);

    Eigen::RowVectorXd forward_ret(X_tmp.rows());
    forward_ret = X_tmp * kernel.transpose();

    double loss_part1 = forward_ret.transpose().dot(Y.col(0));
    double loss_part2 = log((exp(forward_ret.array()) + 1.0).array()).sum();
    double loss = loss_part1 - loss_part2;

    return loss;
}

Eigen::RowVectorXd logisticRegression::compute_kernel_gradient(const Eigen::MatrixXd &X,
                                                               const Eigen::MatrixXd &Y) {
    Eigen::RowVectorXd kernel_gradient(X.cols());
    double loss = compute_loss(X, Y);
    Eigen::RowVectorXd forward_ret = sigmoid(X);

    for (auto i = 0; i < X.cols(); ++i) {
        kernel_gradient(0, i) = Y.col(0).dot(X.col(i)) + (forward_ret.array() - 1.0).matrix().dot(X.col(i));
    }
    return kernel_gradient * loss;
}

void logisticRegression::forward_backward_update(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    Eigen::RowVectorXd kernel_gradient = compute_kernel_gradient(X, Y);
    kernel = kernel - lr * kernel_gradient;
}

