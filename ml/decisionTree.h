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
              size_t split_feats_index_, double split_feats_value,
              std::vector<size_t > &split_feats_index_vec_,
              std::vector<size_t > &feats_row_index_vec_);

    void extend_tree_node(int depth);
    double search_node(const Eigen::RowVectorXd &feats_vec);

    void print_node(int depth);
    double get_node_label() {return node_label;};
    std::map<double, treeNode> get_child() {return child_treeNodes;};
    void set_node_label() {
        if (is_node_need_extend()) {
            return;
        } else {
            node_label = node_origin_labels_sel(0, 0);
        }
    }

private:
    Eigen::MatrixXd node_origin_features; // 原始特征矩阵
    Eigen::MatrixXd node_origin_labels; // 原始标签矩阵
    Eigen::MatrixXd node_origin_disperse_features; // 原始稀疏特征矩阵
    Eigen::MatrixXd node_origin_features_sel; // 按照信息增益选择的原始特征矩阵
    Eigen::MatrixXd node_origin_labels_sel; // 按照信息增益选择的标签矩阵
    Eigen::MatrixXd node_origin_disperse_features_sel; // 按照信息增益选择的稀疏特征矩阵
    std::vector<size_t > feats_has_been_selected_index; // 决策树节点按照信息增益原则选择的特征索引(顺序为从树根到子节点)
    std::vector<size_t > samples_has_been_selected_index; // 决策树节点按照信息增益原则选择的样本索引
    size_t split_feats_index = WRONG_SPLIT_FEATS_INDEX; // 节点分裂的特征索引
    double split_feats_value = WRONG_SPLIT_FEATS_VALUE; //　分裂节点对应的特征离散值
    double node_label = WRONG_NODE_LABEL; // 子节点对应的节点标签
    std::map<double, treeNode> child_treeNodes; // 节点的子节点容器

    // 节点还需要分裂
    bool is_node_need_extend();
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

    treeNode get_dtree() {return decision_tree;};

private:
    // 预测一个实例
    void predict_each_eample(const Eigen::RowVectorXd &X, double *label);

    // 决策树
    treeNode decision_tree;

    // ID3算法生成决策树
    void build_ID3_decision_tree(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
};

#endif //MACHINE_LEARNING_PACKAGE_DECISIONTREE_H
