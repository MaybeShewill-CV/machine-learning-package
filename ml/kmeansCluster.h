/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: kmeansCluster.h
* Date: 18-4-18 下午3:26
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_KMEANSCLUSTER_H
#define MACHINE_LEARNING_PACKAGE_KMEANSCLUSTER_H

#include <MLBase.h>

#include <map>
#include <vector>

#include <cluster.h>

class kmeansCluster : public MLBase {
public:
    kmeansCluster();
    ~kmeansCluster() override = default;

    kmeansCluster(int class_nums, double dist_threshold, DISTANCE_TYPE distance_type, int loop_times = 1000);

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;
    std::vector<cluster> get_cluster_vec() {return _cluster_vec;};

    void test();

private:
    int _class_nums = 2;
    int _max_loop_times = 1000;
    double _dist_threshold = 0.0;
    DISTANCE_TYPE _distanceType = EUCLIDEAN;
    std::vector<cluster> _cluster_vec;

    // 预测一个实例
    double predict_one_sample(const Eigen::RowVectorXd &feats);
    // 初始化聚类簇
    void init_kmeans_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    // 计算特征距离矩阵
    Eigen::MatrixXd calculate_distance_matrix(const Eigen::MatrixXd &X);
    // 更新聚类簇
    bool update_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const Eigen::MatrixXd &dist_matrix);
    // 计算特征向量间的距离
    double calculate_distance(const Eigen::RowVectorXd &vectorXd_1, const Eigen::RowVectorXd &vectorXd_2);
};

#endif //MACHINE_LEARNING_PACKAGE_KMEANSCLUSTER_H
