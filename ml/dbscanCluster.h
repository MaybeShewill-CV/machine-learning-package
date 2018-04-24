/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: dbscanCluster.h
* Date: 18-4-23 下午4:26
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_DBSCANCLUSTER_H
#define MACHINE_LEARNING_PACKAGE_DBSCANCLUSTER_H

#include <MLBase.h>
#include <cluster.h>

#include <set>
#include <map>

#include <opencv2/opencv.hpp>

typedef long sample_index;
typedef double label_name;

class dbscanCluster : public MLBase {
public:
    dbscanCluster() = default;
    ~dbscanCluster() override = default;

    dbscanCluster(int core_obj_nums_threshold, double dist_threshold,
                  DISTANCE_TYPE distance_type = EUCLIDEAN);

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;

    std::vector<dbscan_cluster> get_cluster_vec() {return _cluster_vec;};

    void test();

private:
    int _core_obj_nums_threshold = 0;
    double _dist_threshold = 0.0;
    DISTANCE_TYPE _distanceType = EUCLIDEAN;

    std::vector<dbscan_cluster> _cluster_vec;
    // 核心对象索引
    std::map<sample_index, std::set<sample_index > > _core_obj_map;
    // 未访问样本集合
    std::set<sample_index > _no_visit_sample_set;

    // 初始化核心对象集合
    void init_core_obj_set(const Eigen::MatrixXd &X);
    // 检查核心对象集合是否初始化合理
    bool is_core_obj_validly_init(const Eigen::MatrixXd &X);
    // 初始化未访问样本集合
    void init_no_visit_sample_set(const Eigen::MatrixXd &X);
    // 寻找密度可达样本集合
    std::set<sample_index > find_density_reachable_set(const Eigen::MatrixXd &X, sample_index seed_index,
                                                       std::vector<sample_index > sample_has_been_select);
    // 更新核心对象索引和未访问样本集合
    void update_cluster(const std::set<sample_index > &reachable_set);
    // 多数投票发计算聚类簇标签
    double vote_cluster_label(const std::set<sample_index > &sample_set, const Eigen::MatrixXd &Y);
    // 可视化聚类结果
    void vis_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    // 计算特征向量间的距离
    double calculate_distance(const Eigen::RowVectorXd &vectorXd_1, const Eigen::RowVectorXd &vectorXd_2);
    // 预测一个实例
    double predict_one_sample(const Eigen::RowVectorXd &input_vec);

};


#endif //MACHINE_LEARNING_PACKAGE_DBSCANCLUSTER_H
