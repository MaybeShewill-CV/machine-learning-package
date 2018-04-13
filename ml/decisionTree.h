/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: decisionTree.h
* Date: 18-4-12 下午4:23
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_DECISIONTREE_H
#define MACHINE_LEARNING_PACKAGE_DECISIONTREE_H

#include <MLBase.h>

class decisionTree: public MLBase {
public:
    decisionTree() = default;
    ~decisionTree() override = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;

    void test();

private:
    Eigen::MatrixXd sparse_feats_matrix;
    // 按照等值划分方法初始化稀疏特征矩阵
    void init_sparse_feats_matrix(const Eigen::MatrixXd &X);

    // 计算经验熵
    double compute_empirical_entropy(const Eigen::MatrixXd &Y);
    // 计算信息增益
    double compute_information_gain(const Eigen::MatrixXd &Y, int feats_idx);
    // 计算信息增益比
    double compute_information_gain_ratio(const Eigen::MatrixXd &Y, int feats_idx);
};


#endif //MACHINE_LEARNING_PACKAGE_DECISIONTREE_H
