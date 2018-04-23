/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: gmmCluster.h
* Date: 18-4-20 下午7:41
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_GMMCLUSTER_H
#define MACHINE_LEARNING_PACKAGE_GMMCLUSTER_H

#include <MLBase.h>
#include <cluster.h>

class gmmCluster : public MLBase {
public:
    gmmCluster() = default;
    ~gmmCluster() override = default;

    gmmCluster(int class_nums, int max_iter_times, DISTANCE_TYPE distance_type = EUCLIDEAN);

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;

    std::vector<gmm_cluster> get_cluster_vec() {return _cluster_vec;};

private:
    int _class_nums = 2;
    int _max_item_times = 1000;
    std::vector<gmm_cluster> _cluster_vec;
    DISTANCE_TYPE _distanceType = EUCLIDEAN;

    // 预测一个实例
    double predict_one_smaple(const Eigen::RowVectorXd &input);

    // 初始化高斯混合模型
    void init_gmm_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    // 计算每个样本在高斯混合模型中各个成分的后验概率
    Eigen::MatrixXd compute_gmm_posterior_prob_matrix(const Eigen::MatrixXd &X);
    // EM算法更新高斯混合模型
    bool update_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, int iter_time);
    // 多数投票法为聚类簇赋标签
    void vote_cluster_label (const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    // 计算特征向量间的距离
    double calculate_distance(const Eigen::RowVectorXd &vectorXd_1, const Eigen::RowVectorXd &vectorXd_2);

};

#endif //MACHINE_LEARNING_PACKAGE_GMMCLUSTER_H
