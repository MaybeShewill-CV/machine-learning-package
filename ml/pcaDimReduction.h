/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: pcaDimReduction.h
* Date: 18-5-8 下午1:30
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_PCADIMREDUCTION_H
#define MACHINE_LEARNING_PACKAGE_PCADIMREDUCTION_H

#include <MLBase.h>

class pcaDimReduction : public MLBase {
public:
    pcaDimReduction() = default;
    ~pcaDimReduction() override = default;

    explicit pcaDimReduction(int dims);

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;

private:
    int _reduction_dims = 2;
    Eigen::MatrixXd _project_matrix;
    Eigen::VectorXd _mean_vec;

    // 计算均值向量
    void compute_mean_vec(const Eigen::MatrixXd &X) {_mean_vec = X.colwise().mean();};
    // 特征矩阵中心化
    Eigen::MatrixXd matrix_centralization(const Eigen::MatrixXd &X);
    // 计算协方差矩阵
    Eigen::MatrixXd compute_covariance_matrix(const Eigen::MatrixXd &input);
    // 矩阵特征值分解
    void compute_matrix_decomposition(const Eigen::MatrixXd &input, Eigen::MatrixXd &feat_matrix,
                                      Eigen::MatrixXd &feat_vecs);
    // 构建投影矩阵
    void construct_project_matrix(const Eigen::MatrixXd &feat_vecs, const Eigen::MatrixXd &feat_matrix);
};


#endif //MACHINE_LEARNING_PACKAGE_PCADIMREDUCTION_H
