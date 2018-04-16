/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: decisionTree.cpp
* Date: 18-4-12 下午4:23
************************************************/

#include "decisionTree.h"

#include <set>

#include <glog/logging.h>

#include <globalUtils.h>

#define DEBUG

void decisionTree::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    build_ID3_decision_tree(X, Y);

#ifdef DEBUG
    decision_tree.print_node(0);
#endif
}

void decisionTree::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::MatrixXd ret(X.rows(), 1);
    for (auto i = 0; i < X.rows(); ++i) {
        double label = WRONG_NODE_LABEL;
        predict_each_eample(X.row(i), &label);
        ret(i, 0) = label;
    }
    RET = ret;

#ifdef DEBUG
    LOG(INFO) << "预测结果为: " << RET << std::endl;
#endif
}

void decisionTree::predict_each_eample(const Eigen::RowVectorXd &X, double *label) {
    *label = decision_tree.search_node(X);
}

namespace internal_func {
    int count_elements(const Eigen::MatrixXd &mat, const double ele) {
        return static_cast<int>((mat.array() == ele).count());
    }
    int count_elements(const Eigen::RowVectorXd &vec, const double ele) {
        return static_cast<int>((vec.array() == ele).count());
    }
    std::vector<size_t > sub_matrix_index(const Eigen::MatrixXd &input,
                                          const long col_idx, const double feats_val) {
        std::vector<size_t > idx_vec;
        for (auto i = 0; i < input.col(col_idx).rows(); ++i) {
            if (input(i, col_idx) == feats_val) {
                idx_vec.push_back(static_cast<size_t >(i));
            }
        }
        return idx_vec;
    }
    Eigen::MatrixXd sub_matrix(const Eigen::MatrixXd &input, const long col_idx,
                               const std::vector<size_t > &idx_vec) {
        long matrix_cols = 0;
        if (input.cols() <= 1) {
            matrix_cols = 1;
            assert(col_idx == 0);
        } else {
            matrix_cols = input.cols() - 1;
        }
        Eigen::MatrixXd sub_matrix(idx_vec.size(), matrix_cols);

        if (input.cols() <= 1) {
            for (size_t row = 0; row < idx_vec.size(); ++row) {
                sub_matrix(row, 0) = input(idx_vec[row], 0);
            }
        } else {
            for (size_t row = 0; row < idx_vec.size(); ++row) {
                auto sub_col = 0;
                for (auto col = 0; col < input.row(idx_vec[row]).cols(); ++col) {
                    if (col == col_idx) {
                        continue;
                    } else {
                        sub_matrix(row, sub_col) = input(idx_vec[row], col);
                        sub_col++;
                    }
                }
            }
        }

        return sub_matrix;
    }
}

void decisionTree::build_ID3_decision_tree(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 初始化树根节点
    treeNode tree_root(X, Y, WRONG_SPLIT_FEATS_INDEX);

    // 递归构造决策树
    tree_root.extend_tree_node();
    decision_tree = tree_root;
}

double treeNode::compute_empirical_entropy(const Eigen::MatrixXd &Y) {
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

double treeNode::compute_information_gain(const Eigen::MatrixXd &Y, const int feats_idx) {
    std::vector<Eigen::MatrixXd> Y_split_vec;
    Eigen::RowVectorXd feats_vec = node_disperse_features.col(feats_idx);
    auto disperse_feats_category_nums = feats_vec.maxCoeff();
    for (auto i = 0; i < disperse_feats_category_nums + 1; ++i) {
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

double treeNode::compute_information_gain_ratio(const Eigen::MatrixXd &Y, int feats_idx) {
    double information_gain = compute_information_gain(Y, feats_idx);
    double empirical_entropy = compute_empirical_entropy(node_disperse_features.col(feats_idx));
    return information_gain / empirical_entropy;
}

treeNode::treeNode(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
                   size_t split_feats_index_) {
    node_origin_features = X;
    node_labels = Y;
    split_feats_index = split_feats_index_;
    // 按照等值划分方法初始化稀疏特征矩阵
    Eigen::MatrixXd disperse_feats_matrix_tmp(X.rows(), X.cols());
    for (auto i = 0; i < X.rows(); ++i) {
        for (auto j = 0; j < X.cols(); ++j) {
            disperse_feats_matrix_tmp(i, j) = std::floor(X(i, j));
        }
    }
    node_disperse_features = disperse_feats_matrix_tmp;
}

void treeNode::extend_tree_node() {
    // 根据该节点的特征信息增益选择需要分裂的特征点
    std::vector<double> info_gain_scores;
    for (auto i = 0; i < node_origin_features.cols(); ++i) {
        info_gain_scores.push_back(compute_information_gain(node_labels, i));
    }
    std::vector<size_t > sort_idx = GlobalUtils::sort_indexes(info_gain_scores);
    size_t idx = sort_idx[sort_idx.size() - 1];
    if (idx >= split_feats_index) {
        split_feats_index = idx + 1;
    } else {
        split_feats_index = idx;
    }
    std::map<double, treeNode> child_nodes_tmp;
    std::set<double> disperse_feats_values;
    for (auto i = 0; i < node_disperse_features.col(idx).rows(); ++i) {
        disperse_feats_values.insert(node_disperse_features(i, idx));
    }

    for (auto feats_val : disperse_feats_values) {
        // 选择特征值等于feats_val的行号
        std::vector<size_t > row_idx = internal_func::sub_matrix_index(node_disperse_features, idx, feats_val);

        // 选择特征值等于feats_val的除去该特征列的其余特征矩阵
        Eigen::MatrixXd node_origin_features_tmp = internal_func::sub_matrix(node_origin_features, idx, row_idx);
        Eigen::MatrixXd node_labels_tmp = internal_func::sub_matrix(node_labels, 0, row_idx);
        child_nodes_tmp.insert(std::make_pair(feats_val,
                                              treeNode(node_origin_features_tmp, node_labels_tmp,
                                                       split_feats_index)));
    }

    for (auto &child_node : child_nodes_tmp) {
        if (child_node.second.is_node_need_extend()) {
            child_node.second.extend_tree_node();
            child_node.second.set_node_label();
        } else {
            child_node.second.set_node_label();
        }
    }

    child_treeNodes = child_nodes_tmp;
}

double treeNode::search_node(const Eigen::RowVectorXd &feats_vec) {
    auto col_idx = split_feats_index;
    auto feats_value = feats_vec(0, col_idx);
    feats_value = std::floor(feats_value);
    auto child_node = child_treeNodes.find(feats_value);
    if (child_node == child_treeNodes.end()) {
        LOG(INFO) << "子节点" << split_feats_index << "特征空间中没有找到对应的稀疏特征值： " << feats_value << std::endl;
        return WRONG_NODE_LABEL;
    }

    if (child_node->second.is_node_need_extend()) {
        return child_node->second.search_node(feats_vec);
    }
    else {
        return child_node->second.get_node_label();
    }
}

bool treeNode::is_node_need_extend() {
    // 如果特征空间已经分块完毕则不继续生长子叶子
    if (node_origin_features.cols() <= 1) {return false;}

    // 如果该节点全部的类别一致则不继续生长
    std::set<double> label_set;
    for (auto i = 0; i < node_labels.rows(); ++i) {
        label_set.insert(node_labels(i, 0));
    }
    return label_set.size() > 1;
}

void treeNode::print_node(const int depth) {
    if (is_node_need_extend()) {
        LOG(INFO) << "第" << depth <<  " 层需要分裂，分裂特征索引为: " << split_feats_index
                  << " 子节点个数为: " << child_treeNodes.size() << std::endl;
        for (auto &node_tmp : child_treeNodes) {
            int d = depth + 1;
            node_tmp.second.print_node(d);
        }
    } else {
        LOG(INFO) << "到达第" << depth << "层叶子节点, 标签值为: " << node_label << std::endl;
        return;
    }
}

void decisionTree::test() {
    Eigen::MatrixXd X(4, 15);
    Eigen::MatrixXd Y(15, 1);
    X << 0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,1,2,2,
            2,1,1,2,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0;
    X.transposeInPlace();
    Y << 0,0,1,1,0,0,0,1,1,1,1,1,1,1,0;

    fit(X, Y);
    Eigen::MatrixXd predict_ret;
    predict(X, predict_ret);
}