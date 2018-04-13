/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: decisionTree.cpp
* Date: 18-4-12 下午4:23
************************************************/

#include "decisionTree.h"

#include <map>

#include <glog/logging.h>

#include <globalUtils.h>

void decisionTree::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {

}

void decisionTree::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {

}

namespace internal_func {
    int count_elements(const Eigen::MatrixXd &mat, const double ele) {
        return static_cast<int>((mat.array() == ele).count());
    }
    int count_elements(const Eigen::RowVectorXd &vec, const double ele) {
        return static_cast<int>((vec.array() == ele).count());
    }
}

void decisionTree::init_sparse_feats_matrix(const Eigen::MatrixXd &X) {
    Eigen::MatrixXd sparse_feats_matrix_tmp(X.rows(), X.cols());
    for (auto i = 0; i < X.rows(); ++i) {
        for (auto j = 0; j < X.cols(); ++j) {
            sparse_feats_matrix_tmp(i, j) = std::floor(X(i, j));
        }
    }
    sparse_feats_matrix = sparse_feats_matrix_tmp;
}

double decisionTree::compute_empirical_entropy(const Eigen::MatrixXd &Y) {
    std::map<int, int> label_instance_count;
    for (auto i = 0; i < Y.rows(); ++i) {
        auto label = static_cast<int>(Y(i, 0));
        if (GlobalUtils::has_key(label_instance_count, label)) {
            label_instance_count[label] += 1;
        }
        else {
            label_instance_count.insert(std::make_pair(label, 1));
        }
    }

    auto label_nums = static_cast<double>(Y.rows());
    double empirical_entropy = 0.0;
    for (auto &key_value : label_instance_count) {
        auto tmp = key_value.second / label_nums;
        empirical_entropy += -tmp * std::log(tmp) / std::log(2);
    }
    return empirical_entropy;
}

double decisionTree::compute_information_gain(const Eigen::MatrixXd &Y, const int feats_idx) {
    std::vector<Eigen::MatrixXd> Y_split_vec;
    Eigen::RowVectorXd feats_vec = sparse_feats_matrix.col(feats_idx);
    auto sparse_feats_category_nums = feats_vec.maxCoeff();
    for (auto i = 0; i < sparse_feats_category_nums + 1; ++i) {
        Eigen::MatrixXd Y_split(internal_func::count_elements(feats_vec, static_cast<double>(i)), 1);
        int row_index = 0;
        for (auto j = 0; j < feats_vec.cols(); ++j) {
            if (feats_vec(0, j) == i) {
                Y_split(row_index, 0) = Y(j, 0);
                row_index++;
            }
        }
        Y_split_vec.push_back(Y_split);
    }

    auto sample_nums = static_cast<double>(Y.rows());
    double empirical_entropy  = compute_empirical_entropy(Y);
    for (auto &Y_split : Y_split_vec) {
        auto sample_nums_select = static_cast<double>(Y_split.rows());
        auto empirical_conditional_entropy = compute_empirical_entropy(Y_split);
        empirical_entropy -= sample_nums_select * empirical_conditional_entropy / sample_nums;
    }

    return empirical_entropy;
}

double decisionTree::compute_information_gain_ratio(const Eigen::MatrixXd &Y, int feats_idx) {
    double information_gain = compute_information_gain(Y, feats_idx);
    double empirical_entropy = compute_empirical_entropy(sparse_feats_matrix.col(feats_idx));
    return information_gain / empirical_entropy;
}

void decisionTree::test() {
    Eigen::MatrixXd X(15, 1);
    Eigen::MatrixXd Y(15, 1);
    X << 0.1,0.1,0.2,0.4,0,1,1,1,1.7,1,2,2,2,2,2;
    Y << 0,0,1,1,0,0,0,1,1,1,1,1,1,1,0;

    init_sparse_feats_matrix(X);

    double entropy_gain = compute_information_gain(Y, 0);
    double entropy_gain_ratio = compute_information_gain_ratio(Y, 0);
    LOG(INFO) << "信息增益为: " << entropy_gain << std::endl;
    LOG(INFO) << "信息增益比为: " << entropy_gain_ratio << std::endl;
}