/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: linearRegression.cpp
* Date: 18-4-3 下午3:00
************************************************/

#include "linearRegression.h"

#include <iomanip>
#include <cstdlib>

#include <glog/logging.h>

//#define DEBUG

LinearRegression::LinearRegression() = default;

LinearRegression::~LinearRegression() = default;

LinearRegression::LinearRegression(const double lr, const int iter_nums):
    lr(lr), iter_nums(iter_nums) {}

void LinearRegression::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    init_weights(static_cast<int>(X.cols()), static_cast<int>(Y.cols()), kernel);
    bias = init_bias();

    LOG(INFO) << "Training Begins:" << std::endl;
    for (auto iter = 0; iter < iter_nums; ++iter) {
        auto preds = linear_transform(X, kernel, bias);
        auto loss = compute_square_loss(Y, preds);
        auto kernel_gradient = compuet_kernel_gradient(preds, Y, X);
        auto bias_gradient = compuet_bias_gradient(preds, Y);

#ifdef DEBUG
        LOG(INFO) << "Kernel gradient is: " << kernel_gradient
                  << std::endl << "Bias gradient is: " << bias_gradient << std::endl;
#endif

        // 更新权重
        kernel = kernel - lr * kernel_gradient;
        bias = bias - lr * bias_gradient;

        if (std::abs(loss) < 0.000000001) {
            LOG(INFO) << "迭代完毕" << std::endl;
            break;
        }

        LOG(INFO) << "Iter Step: " << std::setw(4) << iter << " Loss= "
                  << std::setprecision(7) << std::showpoint << loss << std::endl;
    }
}

void LinearRegression::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    RET = linear_transform(X, kernel, bias);
}

void LinearRegression::init_weights(const int feats_dims, const int label_dims, Eigen::MatrixXd &kernel) {
    kernel = Eigen::MatrixXd::Random(feats_dims, label_dims);
}

double LinearRegression::init_bias() {
    return random() / double(RAND_MAX);
}

Eigen::MatrixXd LinearRegression::linear_transform(const Eigen::MatrixXd &X,
        const Eigen::MatrixXd &kernal,
        const double bias) {
    auto x_cols = X.cols();
    auto kernel_rows = kernal.rows();
    if (x_cols != kernel_rows) {
        LOG(ERROR) << "Matrix Multiplication shape error";
        assert(x_cols == kernel_rows);
    }
    Eigen::MatrixXd ret = X * kernal;
    ret = ret.array() + bias;
#ifdef DEBUG
    LOG(INFO) << "Input matrix is: " << std::endl << X << std::endl
               << "Kernel is: " << std::endl << kernal << std::endl << "Bias is: " << bias << std::endl;
#endif
    return ret;
}

double LinearRegression::compute_square_loss(const Eigen::MatrixXd &label,
        const Eigen::MatrixXd &predicts) {
    auto diff = label - predicts;
    diff.array().pow(2);
    return diff.sum() * 0.5;
}

Eigen::MatrixXd LinearRegression::compuet_kernel_gradient(const Eigen::MatrixXd &preds,
        const Eigen::MatrixXd &label,
        const Eigen::MatrixXd &X) {
    auto diff = label - preds;
    Eigen::MatrixXd gradient(X.cols(), 1);
    for (auto i = 0; i < X.cols(); ++i) {
        gradient(i, 0) = diff.col(0).dot(X.col(i)) * (-1);
    }
    return gradient;
}

double LinearRegression::compuet_bias_gradient(const Eigen::MatrixXd &preds,
        const Eigen::MatrixXd &label) {
    auto diff = label - preds;
    return diff.sum() * (-1);
}
