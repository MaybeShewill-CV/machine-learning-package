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
//    LOG(INFO) << "预测结果为: " << RET << std::endl;
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
    // 按照col_idx_vec, row_idx_vec所包含的行列号选择子矩阵
    template <typename INDEX_VAL>
    Eigen::MatrixXd sub_matrix(const Eigen::MatrixXd &input,
                               const std::vector<INDEX_VAL > &row_idx_vec,
                               const std::vector<INDEX_VAL> &col_idx_vec) {
        // 如果行号索引向量为空则取所有的行
        Eigen::MatrixXd matrix_row;
        if (row_idx_vec.empty()) {
            matrix_row = input;
        } else {
            Eigen::MatrixXd matrix_row_tmp(row_idx_vec.size(), input.cols());
            for (size_t i = 0; i < row_idx_vec.size(); ++i) {
                matrix_row_tmp.row(i) = input.row(row_idx_vec[i]);
            }
            matrix_row = matrix_row_tmp;
        }

        // 如果列号索引向量为空则取所有列
        Eigen::MatrixXd matrix_col;
        if (col_idx_vec.empty()) {
            matrix_col = matrix_row;
        } else {
            Eigen::MatrixXd matrix_col_tmp(matrix_row.rows(), col_idx_vec.size());
            for (size_t i = 0; i < col_idx_vec.size(); ++i) {
                matrix_col_tmp.col(i) = matrix_row.col(col_idx_vec[i]);
            }
            matrix_col = matrix_col_tmp;
        }
        return matrix_col;
    }
    // 按照矩阵col_idx列值等于feats_val选择子矩阵行索引向量
    std::vector<size_t > sub_matrix_index(const Eigen::MatrixXd &input,
                                          const long col_idx, const double feats_val,
                                          const std::vector<size_t > &row_idx_vec=std::vector<size_t >()) {
        std::vector<size_t > idx_vec;
        // 如果row_idx_vec为空则搜索input的全部行
        if (row_idx_vec.empty()) {
            for (auto i = 0; i < input.col(col_idx).rows(); ++i) {
                if (input(i, col_idx) == feats_val) {
                    idx_vec.push_back(static_cast<size_t >(i));
                }
            }
        }
        // 否则只搜索行索引向量row_idx_vec中的行
        else {
            for (auto &row : row_idx_vec) {
                if (input(row, col_idx) == feats_val) {
                    idx_vec.push_back(row);
                }
            }

        }

        return idx_vec;
    }
}

void decisionTree::build_ID3_decision_tree(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 初始化树根节点
    std::vector<size_t > feats_has_been_selected;
    std::vector<size_t > samples_has_been_selected;
    treeNode tree_root(X, Y, WRONG_SPLIT_FEATS_INDEX, WRONG_SPLIT_FEATS_VALUE,
                       feats_has_been_selected, samples_has_been_selected);

    // 递归构造决策树
    tree_root.extend_tree_node(0);
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
    Eigen::RowVectorXd feats_vec = node_origin_disperse_features_sel.col(feats_idx);
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
    double empirical_entropy = compute_empirical_entropy(node_origin_disperse_features_sel.col(feats_idx));
    return information_gain / empirical_entropy;
}

treeNode::treeNode(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
                   size_t split_feats_index_, double split_feats_value_,
                   std::vector<size_t > &split_feats_index_vec_,
                   std::vector<size_t > &feats_row_index_vec_) {
    node_origin_features = X;
    node_origin_labels = Y;
    split_feats_index = split_feats_index_;
    split_feats_value = split_feats_value_;
    feats_has_been_selected_index = split_feats_index_vec_;
    samples_has_been_selected_index = feats_row_index_vec_;
    // 按照等值划分方法初始化稀疏特征矩阵
    Eigen::MatrixXd disperse_feats_matrix_tmp(X.rows(), X.cols());
    for (auto i = 0; i < X.rows(); ++i) {
        for (auto j = 0; j < X.cols(); ++j) {
            disperse_feats_matrix_tmp(i, j) = std::floor(X(i, j));
        }
    }
    node_origin_disperse_features = disperse_feats_matrix_tmp;
    std::vector<size_t > feats_remain_index;
    for (auto i = 0; i < node_origin_features.cols(); ++i) {
        if (GlobalUtils::has_elements(feats_has_been_selected_index, static_cast<size_t >(i))) {
            continue;
        } else {
            feats_remain_index.push_back(i);
        }
    }
    node_origin_disperse_features_sel = internal_func::sub_matrix(
            disperse_feats_matrix_tmp, samples_has_been_selected_index,
            feats_remain_index);
    node_origin_features_sel = internal_func::sub_matrix(node_origin_features,
                                                         samples_has_been_selected_index,
                                                         feats_remain_index);
    std::vector<size_t > tmp = {0};
    node_origin_labels_sel = internal_func::sub_matrix(Y, samples_has_been_selected_index, tmp);
}

void treeNode::extend_tree_node(int depth) {
    // 根据该节点的特征信息增益选择需要分裂的特征点
    std::vector<double> info_gain_scores;
    int j = 0;
    for (auto i = 0; i < node_origin_features.cols(); ++i) {
        if (GlobalUtils::has_elements(feats_has_been_selected_index, static_cast<size_t >(i))) {
            info_gain_scores.push_back(-1.0);
        } else {
            info_gain_scores.push_back(compute_information_gain(node_origin_labels_sel, j));
            j++;
        }
    }
    std::vector<size_t > sort_idx = GlobalUtils::sort_indexes(info_gain_scores);
    size_t idx = sort_idx[sort_idx.size() - 1];
    split_feats_index = idx;
//    LOG(INFO) << "第" << depth << "层，选择特征 " << split_feats_index
//              << ",信息增益为: " << info_gain_scores[idx] << std::endl;
//    LOG(INFO) << "第" << depth << "层，稀疏特征矩阵为: " << node_origin_disperse_features_sel << std::endl;
    feats_has_been_selected_index.push_back(split_feats_index);
    std::map<double, treeNode> child_nodes_tmp;
    std::set<double> disperse_feats_values;
    for (auto i = 0; i < node_origin_disperse_features.col(idx).rows(); ++i) {
        disperse_feats_values.insert(node_origin_disperse_features(i, idx));
    }

    for (auto feats_val : disperse_feats_values) {
        // 选择特征值等于feats_val的行号
        std::vector<size_t > row_idx =
                internal_func::sub_matrix_index(node_origin_disperse_features,
                                                idx, feats_val, samples_has_been_selected_index);
        // 选择特征值等于feats_val的除去该特征列的其余特征矩阵
        treeNode child_node(node_origin_features, node_origin_labels, split_feats_index, feats_val,
        feats_has_been_selected_index, row_idx);
        child_nodes_tmp.insert(std::make_pair(feats_val, child_node));
    }

    for (auto &child_node : child_nodes_tmp) {
        if (child_node.second.is_node_need_extend()) {
            child_node.second.extend_tree_node(depth + 1);
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
    if (node_origin_features_sel.cols() <= 1) {return false;}

    // 如果节点有且只有一个样本
//    if (samples_has_been_selected_index.size() <= 1) {return false;}

    // 如果该节点全部的类别一致则不继续生长
    std::set<double> label_set;
    for (auto i = 0; i < node_origin_labels_sel.rows(); ++i) {
        label_set.insert(node_origin_labels_sel(i, 0));
    }
    return label_set.size() > 1;
}

void treeNode::print_node(const int depth) {
    // TODO 需要检查分裂的时候有没有特征选择错误,然后寻找更好的可视化树结构的方法
    if (is_node_need_extend()) {
        LOG(INFO) << "第" << depth <<  "层需要分裂，分裂特征索引为: " << split_feats_index
                  << "，分裂节点特征值为: " << split_feats_value
                  << " 子节点个数为: " << child_treeNodes.size() << std::endl;
        for (auto &node_tmp : child_treeNodes) {
            int d = depth + 1;
            node_tmp.second.print_node(d);
        }
    } else {
        LOG(INFO) << "到达第" << depth << "层叶子节点, 分裂特征索引为: " << split_feats_index
                  << "，分裂节点特征值为: " << split_feats_value
                  << "，标签值为: " << node_label << std::endl;
        return;
    }
}
