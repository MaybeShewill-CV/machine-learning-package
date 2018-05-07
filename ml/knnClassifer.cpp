/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: knnClassifer.cpp
* Date: 18-4-8 下午6:24
************************************************/

#include "knnClassifer.h"

#include <vector>

#include <glog/logging.h>

#include <globalUtils.h>

//#define DEBUG

KnnClassifer::KnnClassifer(): distance_type(EUCLIDEAN) {};

KnnClassifer::~KnnClassifer() = default;

KnnClassifer::KnnClassifer(int k_nums, DISTANCE_TYPE distance_type):
        k_nums(k_nums), distance_type(distance_type) {
}

void KnnClassifer::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    auto feats_dims = X.cols();

#ifdef DEBUG
    LOG(INFO) << "输入特征矩阵为: " << X << std::endl;
    LOG(INFO) << "输入标签为: " << Y << std::endl;
#endif

    // 初始化归一化参数
    Eigen::RowVectorXd norm_parameter_tmp(feats_dims);
    for (auto i = 0; i < feats_dims; ++i) {
        norm_parameter_tmp(i) = X.col(i).maxCoeff();
    }
    norm_parameters = norm_parameter_tmp;

#ifdef DEBUG
    LOG(INFO) << "归一化参数向量tmp为: " << norm_parameter_tmp << std::endl;
    LOG(INFO) << "归一化参数向量为: " << norm_parameters << std::endl;
#endif

    Eigen::MatrixXd norm_feats_tmp(X.rows(), X.cols());
    for (auto i = 0; i < X.rows(); ++i) {
        norm_feats_tmp.row(i) = X.row(i).array() / norm_parameters.array();
    }
    norm_feats = norm_feats_tmp;
    label = Y;

#ifdef DEBUG
    LOG(INFO) << "归一化特征矩阵tmp为: " << norm_feats_tmp << std::endl;
    LOG(INFO) << "归一化特征矩阵为: " << norm_feats << std::endl;
#endif
}

void KnnClassifer::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::MatrixXd norm_feats_tmp(X.rows(), X.cols());
    for (auto i = 0; i < X.rows(); ++i) {
        norm_feats_tmp.row(i) = X.row(i).array() / norm_parameters.array();
    }

#ifdef DEBUG
    LOG(INFO) << "归一化特征矩阵为: " << norm_feats_tmp << std::endl;
#endif

    // 计算距离
    Eigen::MatrixXd feats_distance_matrix(X.rows(), norm_feats.rows());
    for (auto i = 0; i < norm_feats_tmp.rows(); ++i) {
        for (auto j = 0; j < norm_feats.rows(); ++j) {
            double distance = calculate_distance(norm_feats_tmp.row(i), norm_feats.row(j));
            feats_distance_matrix(i, j) = distance;
        }
    }
#ifdef DEBUG
    LOG(INFO) << "距离矩阵第一行为: " << feats_distance_matrix.row(0) << std::endl
              << "行向量大小为: " << feats_distance_matrix.row(0).cols() << std::endl;
#endif

    // 计算k近邻
    Eigen::MatrixXi knn_label_matrix(X.rows(), k_nums);
    Eigen::MatrixXd knn_distance_matrix(X.rows(), k_nums);
    for (auto i = 0; i < feats_distance_matrix.rows(); ++i) {
        std::vector<double> feats_distance;
        for (auto j = 0; j < feats_distance_matrix.row(i).cols(); ++j) {
            feats_distance.push_back(feats_distance_matrix(i, j));
        };
        auto sort_idx = GlobalUtils::sort_indexes(feats_distance);

        for (auto j = 0; j < k_nums; ++j) {
            knn_label_matrix(i, j) = static_cast<int>(label(sort_idx[j]));
            knn_distance_matrix(i, j) = feats_distance_matrix(i, sort_idx[j]);
        }
    }

#ifdef DEBUG
    LOG(INFO) << "k近邻矩阵为: " << knn_label_matrix << std::endl;
    LOG(INFO) << "k近邻距离矩阵为: " << knn_distance_matrix << std::endl;
#endif

    // knn决策部分 不考虑样本不均衡问题,直接采用最原始的平均投票
    Eigen::MatrixXd knn_votes_matrix(knn_label_matrix.rows(), k_nums);
    for (auto i = 0; i < knn_label_matrix.rows(); ++i) {
        for (auto j = 0; j < knn_label_matrix.row(i).cols(); ++j) {
            knn_votes_matrix(i, j) = (knn_label_matrix.row(i).array() == knn_label_matrix(i, j)).count();
        }
    }

#ifdef DEBUG
    LOG(INFO) << "k近邻投票矩阵为: " << knn_votes_matrix << std::endl;
#endif

    Eigen::MatrixXd knn_top_votes_matrix(X.rows(), 1);
    for (auto i = 0; i < knn_votes_matrix.rows(); ++i) {
        Eigen::DenseIndex tmp;
        knn_votes_matrix.row(i).array().maxCoeff(&tmp);
        knn_top_votes_matrix(i, 0) = knn_label_matrix(i, tmp);
    }

#ifdef DEBUG
    LOG(INFO) << "k近邻最终投票结果为: " << knn_top_votes_matrix << std::endl;
#endif

    RET = knn_top_votes_matrix;
}

double KnnClassifer::calculate_distance(const Eigen::ArrayXXd &array_1, const Eigen::ArrayXXd &array_2) {
    double distance = 0.0;
    switch (distance_type) {
        case DISTANCE_TYPE::EUCLIDEAN: {
            Eigen::ArrayXXd diff = array_1 - array_2;
#ifdef DISDEBUG
            LOG(INFO) << "向量差为: " << diff << std::endl;
#endif
            diff = diff.pow(2);
#ifdef DISDEBUG
            LOG(INFO) << "向量平方为: " << diff << std::endl;
#endif
            distance = std::sqrt(diff.sum());
#ifdef DISDEBUG
            LOG(INFO) << "向量间距离为: " << distance << std::endl;
#endif
            break;
        }
        case DISTANCE_TYPE::COSINE: {
            Eigen::ArrayXXd dot_array = array_1 * array_2;
            double dot = dot_array.sum();
#ifdef DISDEBUG
            LOG(INFO) << "向量点积为: " << dot << std::endl;
#endif
            double array_1_length = std::sqrt(array_1.pow(2).sum());
#ifdef DISDEBUG
            LOG(INFO) << "向量一模长为: " << array_1_length << std::endl;
#endif
            double array_2_length = std::sqrt(array_2.pow(2).sum());
#ifdef DISDEBUG
            LOG(INFO) << "向量二模长为: " << array_2_length << std::endl;
#endif
            distance = dot / (array_1_length * array_2_length);
#ifdef DISDEBUG
            LOG(INFO) << "向量间距离为: " << distance << std::endl;
#endif
            break;
        }
    }
    return distance;
}

