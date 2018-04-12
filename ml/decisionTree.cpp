/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: decisionTree.cpp
* Date: 18-4-12 下午4:23
************************************************/

#include "decisionTree.h"

#include <map>

#include <globalUtils.h>

void decisionTree::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {

}

void decisionTree::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {

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

    auto label_nums = label_instance_count.size();
    double empirical_entropy = 0.0;
    for (auto &key_value : label_instance_count) {
        auto tmp = key_value.second / label_nums;
        empirical_entropy += -tmp * std::log(tmp) / std::log(2);
    }
    return empirical_entropy;
}

double decisionTree::compute_information_gain(const Eigen::MatrixXd X, const Eigen::MatrixXd &Y,
                                              const int feats_idx) {
    // TODO 按照特征feats_idx的离散取值将Y分后计算每一块的经验增益,然后计算该特征的信息增益

}

