/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: logisticRegression.cpp
* Date: 18-4-10 下午2:39
************************************************/

#include "logisticRegression.h"

#include <glog/logging.h>

//#define DEBUG

logisticRegression::logisticRegression(double learning_rate, int iter_times):
    lr(learning_rate), iter_times(iter_times) {}

void logisticRegression::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 初始化权重
    auto feats_dims = static_cast<int>(X.cols());
    init_kernel(feats_dims);

    // 设置随机数生成引擎
    std::default_random_engine e;
    std::uniform_int_distribution<unsigned> u(0, static_cast<unsigned>(X.rows() - 1));

    // bp更新权重
    LOG(INFO) << "开始训练:" << std::endl;
    for (auto i = 0; i < iter_times; ++i) {
        double loss = compute_loss(X, Y);
        if (std::abs(loss) < 0.0000001) {
            LOG(INFO) << "迭代训练完毕" << std::endl;
            return;
        }

        LOG(INFO) << "Iter: " << i << " Loss: " << loss << std::endl;

        if (gdtype == GDTYPE::SGD) {
            uint random_index = u(e);
            forward_backward_update(X.row(random_index), Y.row(random_index));
#ifdef DEBUG
            Eigen::RowVectorXd gradient = compute_kernel_gradient(X.row(random_index), Y.row(random_index));
            LOG(INFO) << "参数更新权重为: " << gradient << std::endl;
#endif
        } else {
            forward_backward_update(X, Y);
#ifdef DEBUG
            Eigen::RowVectorXd gradient = compute_kernel_gradient(X, Y);
            LOG(INFO) << "参数更新权重为: " << gradient << std::endl;
#endif
        }

    }
}

void logisticRegression::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::RowVectorXd forward_ret = forward(X);
    Eigen::MatrixXd ret_tmp(forward_ret.cols(), 1);
    for (auto i = 0; i < forward_ret.cols(); ++i) {
        ret_tmp(i, 0) = forward_ret(0, i);
    }
    RET = ret_tmp;
}

void logisticRegression::init_kernel(const int feats_dims) {
    kernel = Eigen::RowVectorXd::Random(feats_dims + 1);
}

void logisticRegression::normalize_feats(Eigen::MatrixXd &X) {
    for (auto i = 0; i < X.cols(); ++i) {
        X.col(i).normalize();
    }
}

Eigen::RowVectorXd logisticRegression::sigmoid(const Eigen::RowVectorXd &X) {
    Eigen::RowVectorXd sigmoid_ret = exp(((X.array() * (-1)).array()).array()) + 1;
    return sigmoid_ret.cwiseInverse();
}

Eigen::RowVectorXd logisticRegression::forward(Eigen::MatrixXd X) {
    assert(kernel.cols() == X.cols() + 1);

    Eigen::MatrixXd X_tmp(X.rows(), X.cols() + 1);
    for (auto i = 0 ; i < X.cols(); ++i) {
        X_tmp.col(i) = X.col(i);
    }
    X_tmp.col(X_tmp.cols() - 1) = Eigen::MatrixXd::Ones(X.rows(), 1);
    normalize_feats(X_tmp);

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
    normalize_feats(X_tmp);

    Eigen::RowVectorXd forward_ret(X_tmp.rows());
    forward_ret = X_tmp * kernel.transpose();

    Eigen::RowVectorXd Y_vec(Y.rows());
    for (auto i = 0; i < Y.rows(); ++i) {
        if (Y(i, 0) == 0.0) {
            Y_vec(0, i) = -1.0;
        } else {
            Y_vec(0, i) = 1.0;
        }
    }

    Eigen::RowVectorXd loss_vec = (((forward_ret * Y_vec.asDiagonal()) * (-1)).array().exp() + 1).array().log();
    double loss = loss_vec.sum() / X_tmp.rows();

    return loss;
}

Eigen::RowVectorXd logisticRegression::compute_kernel_gradient(const Eigen::MatrixXd &X,
        const Eigen::MatrixXd &Y) {
    Eigen::MatrixXd X_tmp(X.rows(), X.cols() + 1);
    for (auto i = 0 ; i < X.cols(); ++i) {
        X_tmp.col(i) = X.col(i);
    }
    X_tmp.col(X_tmp.cols() - 1) = Eigen::MatrixXd::Ones(X.rows(), 1);
    normalize_feats(X_tmp);

    Eigen::RowVectorXd Y_vec(Y.rows());
    for (auto i = 0; i < Y.rows(); ++i) {
        if (Y(i, 0) == 0.0) {
            Y_vec(0, i) = -1.0;
        } else {
            Y_vec(0, i) = 1.0;
        }
    }

    Eigen::RowVectorXd forward_ret(X_tmp.rows());
    forward_ret = X_tmp * kernel.transpose();

    Eigen::RowVectorXd exp_vec = ((forward_ret * Y_vec.asDiagonal()) * (-1)).array().exp();
    Eigen::RowVectorXd exp_vec_2 = (exp_vec * Y_vec.asDiagonal() * (-1)) *
            ((exp_vec.array() + 1).cwiseInverse()).matrix().asDiagonal();

    Eigen::RowVectorXd kernel_gradient(X_tmp.cols());
    kernel_gradient = exp_vec_2 * X_tmp;

    return kernel_gradient / X_tmp.rows();
}

void logisticRegression::forward_backward_update(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    Eigen::RowVectorXd kernel_gradient = compute_kernel_gradient(X, Y);
    kernel = kernel - lr * kernel_gradient;
}

