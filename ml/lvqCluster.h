/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: lvqCluster.h
* Date: 18-4-19 下午7:32
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_LVQCLUSTER_H
#define MACHINE_LEARNING_PACKAGE_LVQCLUSTER_H

#include <MLBase.h>

#include <cluster.h>

class lvqCluster : public MLBase {
public:
    lvqCluster() = default;
    ~lvqCluster() override = default;

    lvqCluster(int class_nums, double dist_threshold, DISTANCE_TYPE distance_type,
               int loop_times = 1000, double lr = 0.1);

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;

    void test();

    std::vector<lvq_cluster> get_cluster_vec() {return _cluster_vec;};

private:
    int _class_nums = 2;
    int _max_loop_times = 1000;
    double _dist_threshold = 0.0;
    double _lr = 0.01;
    DISTANCE_TYPE _distanceType = EUCLIDEAN;
    std::vector<lvq_cluster> _cluster_vec;

    // 预测一个实例
    double predict_one_sample(const Eigen::RowVectorXd &feats);
    // 初始化聚类簇
    void init_lvq_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    // 更新聚类簇
    bool update_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    // 计算特征向量间的距离
    double calculate_distance(const Eigen::RowVectorXd &vectorXd_1, const Eigen::RowVectorXd &vectorXd_2);
};

#endif //MACHINE_LEARNING_PACKAGE_LVQCLUSTER_H
