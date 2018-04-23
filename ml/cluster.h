/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: cluster.h
* Date: 18-4-19 下午7:34
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_CLUSTER_H
#define MACHINE_LEARNING_PACKAGE_CLUSTER_H

#include <eigen3/Eigen/Dense>

const double WRONG_LABEL = -9999.0;

enum DISTANCE_TYPE {
    EUCLIDEAN,
    COSINE
};

class cluster {
public:
    cluster() = default;
    ~cluster() = default;

    cluster(const Eigen::MatrixXd &sample_feats_matrix, const Eigen::MatrixXd &label_matrix);
    cluster& operator = (const cluster &other);

    double get_cluster_label() {return _cluster_label;};
    Eigen::RowVectorXd get_cluster_mean_feats_vec() {return _mean_feats_vec;};
    Eigen::MatrixXd get_sample_feats_matrix() {return _sample_feats_matrix;};

    // 更新簇(更新簇标签和簇中心向量)
    bool is_cluster_updated(const Eigen::MatrixXd &sample_feats_matrix, const Eigen::MatrixXd &label_matrix);

private:
    double _cluster_label = WRONG_LABEL;
    Eigen::RowVectorXd _mean_feats_vec;
    Eigen::MatrixXd _sample_feats_matrix;
};

class lvq_cluster {
public:
    lvq_cluster() = default;
    ~lvq_cluster() = default;

    lvq_cluster(const Eigen::RowVectorXd &prototype_vec, double dist_threshold,
                DISTANCE_TYPE distance_type, double label, double lr = 0.1);

    Eigen::RowVectorXd get_prototype_vec() {return _prototype_vec;};
    double get_cluster_label() {return _label;};

    // 更新簇(更新簇标签和簇中心向量)
    bool is_cluster_updated(const Eigen::RowVectorXd &diff_prototype_vec, double label,
                            bool is_similar);

private:
    double _lr = 0.1;
    double _dist_threshold = 0.0;
    double _label = WRONG_LABEL;
    DISTANCE_TYPE _distanceType = EUCLIDEAN;
    Eigen::RowVectorXd _prototype_vec;
};

class gmm_cluster {
public:
    gmm_cluster() = default;
    ~gmm_cluster() = default;

    gmm_cluster(const Eigen::MatrixXd &sample_matrixXd, double gmm_mix_coefficient);

    // 获取高斯混合模型参数
    double get_gmm_mix_coefficient() {return _gmm_mix_coefficient;};
    // 获取高斯模型均值向量
    Eigen::RowVectorXd get_mean_vec() {return _mean_vectorXd;};
    // 计算向量概率密度
    double compute_prob_density(const Eigen::RowVectorXd &input_vec);
    // 更新高斯聚类簇
    bool update_gmm_cluster(const Eigen::RowVectorXd &mean_vectorXd,
                            const Eigen::MatrixXd &covariance_matrix,
                            double gmm_mix_coefficient);
    // 给高斯聚类簇赋标签
    void set_cluster_label(double label) {_label = label;};
    // 获取高斯聚类簇标签
    double get_cluster_label() {return _label;};

private:
    Eigen::MatrixXd _sample_matrixXd;
    Eigen::RowVectorXd _mean_vectorXd;
    Eigen::MatrixXd _covariance_matrixXd;
    double _gmm_mix_coefficient = 1.0;
    double _label = WRONG_LABEL;

    // 计算协方差矩阵
    Eigen::MatrixXd compute_covariance_matrix(const Eigen::MatrixXd &input);

};


#endif //MACHINE_LEARNING_PACKAGE_CLUSTER_H
