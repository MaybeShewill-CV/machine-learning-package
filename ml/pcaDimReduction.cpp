/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: pcaDimReduction.cpp
* Date: 18-5-8 下午1:30
************************************************/

#include "pcaDimReduction.h"

#include <glog/logging.h>

#include <globalUtils.h>

pcaDimReduction::pcaDimReduction(int dims) : _reduction_dims(dims) {}

void pcaDimReduction::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    LOG(INFO) << "特征矩阵为: " << X << std::endl;
    // 计算均值向量
    compute_mean_vec(X);
    LOG(INFO) << "均值向量为: " << _mean_vec << std::endl;
    // 样本中心化
    Eigen::MatrixXd feats_mean_matrix = matrix_centralization(X);
    LOG(INFO) << "中心化特征矩阵为: " << feats_mean_matrix << std::endl;
    // 计算协方差矩阵
    Eigen::MatrixXd covariance_matrix = compute_covariance_matrix(feats_mean_matrix);
    LOG(INFO) << "协方差矩阵为: " << covariance_matrix << std::endl;
    // 协方差矩阵特征值分解
    Eigen::MatrixXd feats_matrix;
    Eigen::MatrixXd feats_vecs;
    compute_matrix_decomposition(covariance_matrix, feats_matrix, feats_vecs);
    LOG(INFO) << "特征矩阵为: " << feats_matrix << std::endl;
    LOG(INFO) << "特征值为: " << feats_vecs << std::endl;
    LOG(INFO) << "检查特征值分解: " << feats_matrix * feats_vecs * feats_matrix.inverse() << std::endl;
    // 构建特征投影矩阵
    construct_project_matrix(feats_vecs, feats_matrix);
    LOG(INFO) << "投影矩阵为: " << _project_matrix << std::endl;
}

void pcaDimReduction::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::MatrixXd centralized_X = matrix_centralization(X);
    RET = centralized_X * _project_matrix;
}

Eigen::MatrixXd pcaDimReduction::matrix_centralization(const Eigen::MatrixXd &X) {
    Eigen::MatrixXd ret(X.rows(), X.cols());
    for (auto i = 0; i < X.rows(); ++i) {
        for (auto j = 0; j < X.cols(); ++j) {
            ret(i, j) = X(i, j) - _mean_vec(j);
        }
    }
    return ret;
}

Eigen::MatrixXd pcaDimReduction::compute_covariance_matrix(const Eigen::MatrixXd &input) {
    // 如果输入是行向量,则返回单位矩阵
     if (input.rows() == 1) {
         auto feat_dims = input.cols();
         Eigen::MatrixXd covMat = Eigen::MatrixXd::Identity(feat_dims, feat_dims).array() / 10.0;
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

void pcaDimReduction::compute_matrix_decomposition(const Eigen::MatrixXd &input,
                                                   Eigen::MatrixXd &feat_matrix,
                                                   Eigen::MatrixXd &feat_vecs) {
    Eigen::EigenSolver<Eigen::MatrixXd> es(input);

    feat_vecs = es.pseudoEigenvalueMatrix();
    feat_matrix = es.pseudoEigenvectors();
}

void pcaDimReduction::construct_project_matrix(const Eigen::MatrixXd &feat_vecs,
                                               const Eigen::MatrixXd &feat_matrix) {
    // 特征值排序
    std::vector<double> feats_values;
    for (auto i = 0; i < feat_vecs.rows(); ++i) {
        feats_values.push_back(feat_vecs(i, i));
    }
    std::vector<size_t > sort_index = GlobalUtils::sort_indexes(feats_values);

    // 按照排序特征值选择特征向量构建投影矩阵
    Eigen::MatrixXd project_matrix_tmp(feat_matrix.rows(), _reduction_dims);
    for (auto i = 0; i < _reduction_dims; ++i) {
        project_matrix_tmp.col(i) = feat_matrix.col(sort_index[sort_index.size() - 1 - i]);
    }

    _project_matrix = project_matrix_tmp;
}
