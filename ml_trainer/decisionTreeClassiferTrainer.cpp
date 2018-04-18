/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: decisionTreeClassiferTrainer.cpp
* Date: 18-4-16 下午7:31
************************************************/

#include "decisionTreeClassiferTrainer.h"

#include <glog/logging.h>

//#define DEBUG

decisionTreeClassiferTrainer::decisionTreeClassiferTrainer(DTREE_TYPE dtree_type): dtreeType(dtree_type) {};

void decisionTreeClassiferTrainer::train(const std::string &input_file_path) {
    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

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

    classifier.fit(X, Y);
}

void decisionTreeClassiferTrainer::test(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(INFO) << "模型未训练" << std::endl;
        return;
    }

    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.rows(), input_data.cols() - 1);
    Eigen::MatrixXd Y(input_data.rows(), 1);
    for (auto i  = 0; i < X.cols(); ++i) {
        X.col(i) = input_data.col(i);
    }
    Y = input_data.col(input_data.cols() - 1);

    Eigen::MatrixXd preds;
    classifier.predict(X, preds);
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

void decisionTreeClassiferTrainer::deploy(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(INFO) << "模型未训练" << std::endl;
        return;
    }

    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);
    Eigen::MatrixXd X(input_data.rows(), input_data.cols() - 1);
    for (auto i  = 0; i < X.cols(); ++i) {
        X.col(i) = input_data.col(i);
    }

    Eigen::MatrixXd RET;
    classifier.predict(X, RET);

    LOG(INFO) << "预测结果如下:" << std::endl;
    for (auto i = 0; i < RET.rows(); ++i) {
        LOG(INFO) << "样本: " << i << "　类别: " << RET(i, 0) << std::endl;
    }
}

bool decisionTreeClassiferTrainer::is_model_trained() {
    return !classifier.get_dtree().get_child().empty();
}