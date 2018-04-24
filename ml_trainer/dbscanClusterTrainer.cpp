/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: dbscanClusterTrainer.cpp
* Date: 18-4-24 下午2:19
************************************************/

#include "dbscanClusterTrainer.h"

#include <glog/logging.h>

dbscanClusterTrainer::dbscanClusterTrainer(int core_obj_nums_threshold,
                                           double dist_threshold,
                                           DISTANCE_TYPE distancetype) :
        __core_obj_nums_threshold(core_obj_nums_threshold),
        _dist_threshold(dist_threshold), _distanceType(distancetype){
    _cluster = dbscanCluster(__core_obj_nums_threshold, _dist_threshold, _distanceType);
}

void dbscanClusterTrainer::train(const std::string &input_file_path) {
    Eigen::MatrixXd input_data;
    _dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.rows(), input_data.cols() - 1);
    Eigen::MatrixXd Y(input_data.rows(), 1);
    for (auto i  = 0; i < X.cols(); ++i) {
        X.col(i) = input_data.col(i);
    }
    Y = input_data.col(input_data.cols() - 1);

#ifdef DEBUG
    LOG(INFO) << "特征矩阵为: " << X << std::endl;
    LOG(INFO) << "对应标签为: " << Y << std::endl;
#endif

    _cluster.fit(X, Y);
}

void dbscanClusterTrainer::test(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(INFO) << "模型未训练" << std::endl;
        return;
    }

    Eigen::MatrixXd input_data;
    _dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.rows(), input_data.cols() - 1);
    Eigen::MatrixXd Y(input_data.rows(), 1);
    for (auto i  = 0; i < X.cols(); ++i) {
        X.col(i) = input_data.col(i);
    }
    Y = input_data.col(input_data.cols() - 1);

    Eigen::MatrixXd preds;
    _cluster.predict(X, preds);
    LOG(INFO) << "测试结果如下" << std::endl;
    LOG(INFO) << "Label: ---- Predict: ----" << std::endl;
    int correct_preds_count = 0;
    for (auto i = 0; i < preds.rows(); ++i) {
        LOG(INFO) << Y(i, 0) << " --- " << static_cast<int>(round(preds(i, 0))) << std::endl;
        if (Y(i, 0) == round(preds(i, 0))) {
            correct_preds_count++;
        }
    }
    LOG(INFO) << "测试完毕, 准确率为: " << correct_preds_count * 100 / preds.rows() << "%" << std::endl;
}

void dbscanClusterTrainer::deploy(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(INFO) << "模型未训练" << std::endl;
        return;
    }

    Eigen::MatrixXd input_data;
    _dataLoder.load_data_from_txt(input_file_path, input_data);
    Eigen::MatrixXd X(input_data.rows(), input_data.cols() - 1);
    for (auto i  = 0; i < X.cols(); ++i) {
        X.col(i) = input_data.col(i);
    }

    Eigen::MatrixXd RET;
    _cluster.predict(X, RET);

    LOG(INFO) << "预测结果如下:" << std::endl;
    for (auto i = 0; i < RET.rows(); ++i) {
        LOG(INFO) << "样本: " << i << "　类别: " << RET(i, 0) << std::endl;
    }

}

bool dbscanClusterTrainer::is_model_trained() {
    return !_cluster.get_cluster_vec().empty();
}