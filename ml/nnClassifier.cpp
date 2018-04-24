/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: nnclassifier.cpp
* Date: 18-4-24 下午3:01
************************************************/

#include "nnclassifier.h"

nnclassifier::nnclassifier(const int class_nums, const int layer_nums,
                           const int hidden_uint_nums, ACTIVATION_TYPE activation_type) :
        _class_nums(class_nums), _layer_nums(layer_nums),
        _hidden_unit_nums(hidden_uint_nums){}

void nnclassifier::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {

}

void nnclassifier::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {

}

void nnclassifier::init_kernel(const Eigen::MatrixXd &X) {
    int feats_dims = static_cast<int>(X.cols()) + 1;

    // 初始化参数尺寸
    for (auto i = 0; i < _layer_nums; ++i) {
        if (i == 0) {
            _kernel_size_vec.emplace_back(std::make_pair(feats_dims + 1,
                                                         _hidden_unit_nums));
        } else {
            _kernel_size_vec.emplace_back(std::make_pair(_hidden_unit_nums + 1,
                                                         _hidden_unit_nums));
        }
    }
    _kernel_size_vec.emplace_back(std::make_pair(_hidden_unit_nums + 1, _class_nums));

    // 初始化参数
    for (auto i = 0; i <= _layer_nums; ++i) {
        Eigen::MatrixXd kernel = Eigen::MatrixXd::Random(_kernel_size_vec[i].first,
                                                         _kernel_size_vec[i].second);
        _kernel_vec.push_back(kernel);
    }
}

Eigen::MatrixXd nnclassifier::forward_broadcast(const Eigen::MatrixXd &X) {
    Eigen::MatrixXd input = expand_feats_matrix(X);
    Eigen::MatrixXd ret;
    for (auto i = 0; i <= _layer_nums; ++i) {
        ret = expand_feats_matrix(input * _kernel_vec[i]);
    }
    return ret;
}

Eigen::MatrixXd nnclassifier::expand_feats_matrix(const Eigen::MatrixXd &X) {
    Eigen::MatrixXd ret = Eigen::Matrix::Ones(X.rows(), X.cols() + 1);
    for (auto i = 0; i < X.cols(); ++i) {
        ret.col(i) = X.col(i);
    }
    return ret;
}

double nnclassifier::activate_unit(double input) {
    double out_put = INFINITY;
    switch (_activationType) {
        case RELU: {
            if (input >= 0) {
                out_put = input;
            } else {
                out_put = 0;
            }
            break;
        }
        case SIGMOID: {
            out_put = 1 / (1 + std::exp(-input));
            break;
        }
    }
    return out_put;
}