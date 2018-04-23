/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: cluster.cpp
* Date: 18-4-19 下午7:34
************************************************/

#include "cluster.h"

#include <map>
#include <glog/logging.h>

#include <globalUtils.h>

cluster& cluster::operator=(const cluster &other) {
    // 检查自赋值
    if (this != &other) {
        this->_cluster_label = other._cluster_label;
        // 复制特征矩阵
        this->_sample_feats_matrix = other._sample_feats_matrix;
        // 复制均值向量
        this->_mean_feats_vec = other._mean_feats_vec;
    }
    return *this;
}

cluster::cluster(const Eigen::MatrixXd &sample_feats_matrix, const Eigen::MatrixXd &label_matrix) :
        _sample_feats_matrix(sample_feats_matrix) {
    // 初始化中心向量
    Eigen::RowVectorXd mean_feats_vec(1, sample_feats_matrix.cols());
    for (auto i = 0; i < sample_feats_matrix.cols(); ++i) {
        mean_feats_vec(0, i) = sample_feats_matrix.col(i).mean();
    }
    _mean_feats_vec = mean_feats_vec;

    // 初始化簇标签，选择标签矩阵的多数类
    typedef double class_name;
    std::map<class_name, int> label_count;
    for (auto i = 0; i < label_matrix.rows(); ++i) {
        if (GlobalUtils::has_key(label_count, label_matrix(i, 0))) {
            label_count[label_matrix(i, 0)] += 1;
        } else {
            label_count[label_matrix(i, 0)] = 1;
        }
    }
    class_name cluster_label = WRONG_LABEL;
    int label_nums = -1;
    for (auto &label : label_count) {
        if (label.second > label_nums) {
            cluster_label = label.first;
        }
    }
    _cluster_label = cluster_label;
}

bool cluster::is_cluster_updated(const Eigen::MatrixXd &sample_feats_matrix,
                                 const Eigen::MatrixXd &label_matrix) {
    if (sample_feats_matrix.rows() == 0) {
        return false;
    }
    // 更新中心向量
    Eigen::RowVectorXd mean_feats_vec(1, sample_feats_matrix.cols());
    for (auto i = 0; i < sample_feats_matrix.cols(); ++i) {
        mean_feats_vec(0, i) = sample_feats_matrix.col(i).mean();
    }

    // TODO 聚类簇停止更新条件需要更改为向量间距离小于阈值
    if (mean_feats_vec == _mean_feats_vec) {
        return false;
    }
    _mean_feats_vec = mean_feats_vec;
    _sample_feats_matrix = sample_feats_matrix;

    // 更新簇标签，选择标签矩阵的多数类
    typedef double class_name;
    std::map<class_name, int> label_count;
    for (auto i = 0; i < label_matrix.rows(); ++i) {
        if (GlobalUtils::has_key(label_count, label_matrix(i, 0))) {
            label_count[label_matrix(i, 0)] += 1;
        } else {
            label_count[label_matrix(i, 0)] = 1;
        }
    }
    class_name cluster_label = WRONG_LABEL;
    int label_nums = -1;
    for (auto &label : label_count) {
        if (label.second > label_nums) {
            cluster_label = label.first;
            label_nums = label.second;
        }
    }
    _cluster_label = cluster_label;

    return true;
}

lvq_cluster::lvq_cluster(const Eigen::RowVectorXd &prototype_vec,
                         double dist_threshold, DISTANCE_TYPE distance_type,
                         double label, double lr) {
    _prototype_vec = prototype_vec;
    _dist_threshold = dist_threshold;
    _distanceType = distance_type;
    _label = label;
    _lr = lr;
}

bool lvq_cluster::is_cluster_updated(const Eigen::RowVectorXd &diff_prototype_vec, const double label,
                                     bool is_similar) {
    Eigen::RowVectorXd updated_prototype_vec;
    // 如果相似则更新标签
    if (is_similar) {
        updated_prototype_vec = _prototype_vec + _lr * diff_prototype_vec;
        _label = label;
    } else {
        updated_prototype_vec = _prototype_vec - _lr * diff_prototype_vec;
    }
    // TODO 更改簇停止更新条件为原型向量间的距离不超过一定阈值
    if (updated_prototype_vec == _prototype_vec) {
        return false;
    } else {
        _prototype_vec = updated_prototype_vec;
        return true;
    }
}

gmm_cluster::gmm_cluster(const Eigen::MatrixXd &sample_matrixXd, double gmm_mix_coeffiecient) :
        _gmm_mix_coefficient(gmm_mix_coeffiecient) {
    // 初始化特征空间
    _sample_matrixXd = sample_matrixXd;
    // 初始化均值向量
    _mean_vectorXd = _sample_matrixXd.colwise().mean();
    // 初始化协方差矩阵
    _covariance_matrixXd = compute_covariance_matrix(_sample_matrixXd);
}

double gmm_cluster::compute_prob_density(const Eigen::RowVectorXd &input_vec) {
    auto sample_nums = _sample_matrixXd.rows();
    auto tmp = (input_vec - _mean_vectorXd) * _covariance_matrixXd.inverse()
               * (input_vec - _mean_vectorXd).transpose();
    assert(tmp.rows() == 1 && tmp.cols() == 1);
    double molecular = std::exp(-0.5 * tmp(0, 0));
    double denominator = std::pow(2 * M_PI, static_cast<double>(sample_nums / 2))
                         * std::sqrt(_covariance_matrixXd.determinant());
    return molecular / denominator;
}

Eigen::MatrixXd gmm_cluster::compute_covariance_matrix(const Eigen::MatrixXd &input) {
    // 如果输入是行向量,则返回单位矩阵
    if (input.rows() == 1) {
        auto feat_dims = input.cols();
        Eigen::MatrixXd covMat = Eigen::MatrixXd::Identity(feat_dims, feat_dims) / 10;
        return covMat;
    }

    Eigen::MatrixXd meanVec = input.colwise().mean();
    Eigen::RowVectorXd meanVecRow(Eigen::RowVectorXd::Map(meanVec.data(),input.cols()));

    Eigen::MatrixXd zeroMeanMat = input;
    zeroMeanMat.rowwise() -= meanVecRow;
    Eigen::MatrixXd covMat;

    if(input.rows()==1) {
        covMat = (zeroMeanMat.adjoint() * zeroMeanMat) / static_cast<double>((input.rows()));
    } else {
        covMat = (zeroMeanMat.adjoint()*zeroMeanMat) / static_cast<double>((input.rows()-1));
    }

    return covMat;
}

bool gmm_cluster::update_gmm_cluster(const Eigen::RowVectorXd &mean_vectorXd,
                                     const Eigen::MatrixXd &covariance_matrixXd,
                                     double gmm_mix_coefficient) {
    if (mean_vectorXd == _mean_vectorXd || covariance_matrixXd == _covariance_matrixXd) {
        return false;
    } else {
        _mean_vectorXd = mean_vectorXd;
        _covariance_matrixXd = covariance_matrixXd;
        _gmm_mix_coefficient = gmm_mix_coefficient;
        return true;
    }
}