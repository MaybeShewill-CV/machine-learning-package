/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: decisionTree.h
* Date: 18-4-12 下午4:23
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_DECISIONTREE_H
#define MACHINE_LEARNING_PACKAGE_DECISIONTREE_H

#include <MLBase.h>

#include <map>
#include <set>

const size_t WRONG_SPLIT_FEATS_INDEX = 999999;
const double WRONG_NODE_LABEL = -99999.0;
const double WRONG_SPLIT_FEATS_VALUE = 9999999.0;

class treeNode {
public:
    treeNode() = default;
    ~treeNode() = default;
    treeNode (const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
              size_t split_point_);

    void extend_tree_node();
    double search_node(const Eigen::RowVectorXd &feats_vec);

    void print_node(int depth);
    size_t get_split_feats_index() {return split_feats_index;};
    double get_node_label() {return node_label;};
    void set_node_label() {
        if (is_node_need_extend()) {
            return;
        } else {
            node_label = node_labels(0, 0);
        }
    }

private:
    Eigen::MatrixXd node_origin_features;
    Eigen::MatrixXd node_disperse_features;
    Eigen::MatrixXd node_labels;
    size_t split_feats_index = WRONG_SPLIT_FEATS_INDEX;
    std::map<double, treeNode> child_treeNodes;
    bool is_node_need_extend();
    double node_label = WRONG_NODE_LABEL;

    // 计算经验熵
    double compute_empirical_entropy(const Eigen::MatrixXd &Y);
    // 计算信息增益
    double compute_information_gain(const Eigen::MatrixXd &Y, int feats_idx);
    // 计算信息增益比
    double compute_information_gain_ratio(const Eigen::MatrixXd &Y, int feats_idx);

};

class decisionTree: public MLBase {
public:
    decisionTree() = default;
    ~decisionTree() override = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;

    void test();

private:
    // 预测一个实例
    void predict_each_eample(const Eigen::RowVectorXd &X, double *label);

    // 决策树
    treeNode decision_tree;

    // ID3算法生成决策树
    void build_ID3_decision_tree(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
};

#endif //MACHINE_LEARNING_PACKAGE_DECISIONTREE_H
