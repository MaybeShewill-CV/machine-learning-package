/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: knnClassiferTrainer.cpp
* Date: 18-4-10 上午10:18
************************************************/

#include "knnClassiferTrainer.h"


void knnClassiferTrainer::train(const std::string &input_file_path) {
    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.rows(), input_data.cols() - 1);
    Eigen::MatrixXd Y(input_data.rows(), 1);
    for (auto i = 0; i < input_data.cols() - 1; ++i) {
        X.col(i) = input_data.col(i);
    }
    Y = input_data.col(input_data.cols() - 1);

    classifer.fit(X, Y);
    LOG(INFO) << "模型训练完毕" << std::endl;
}

void knnClassiferTrainer::test(const std::string &input_file_path) {

    if (!is_model_trained()) {
        LOG(ERROR) << "模型未训练" << std::endl;
        return;
    }

    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.rows(), input_data.cols() - 1);
    Eigen::MatrixXd Y(input_data.rows(), 1);
    for (auto i = 0; i < input_data.cols() - 1; ++i) {
        X.col(i) = input_data.col(i);
    }
    Y = input_data.col(input_data.cols() - 1);

    Eigen::MatrixXd RET;
    classifer.predict(X, RET);

    LOG(INFO) << "测试结果如下:" << std::endl;
    LOG(INFO) << "--- Label: --- Predict: ---" << std::endl;
    int correct_preds_counts = 0;
    for (auto i = 0; i < Y.rows(); ++i) {
        LOG(INFO) << "--- " << Y(i, 0) << " --- " << RET(i, 0) << " ---" << std::endl;
        if (Y(i, 0) == RET(i, 0)) {
            correct_preds_counts++;
        }
    }
    LOG(INFO) << "测试完毕, 准确率为: " << correct_preds_counts * 100 / Y.rows() << "%" << std::endl;
}

void knnClassiferTrainer::deploy(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(ERROR) << "模型未训练" << std::endl;
        return;
    }

    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd RET;
    classifer.predict(input_data, RET);

    LOG(INFO) << "预测结果如下:" << std::endl;
    for (auto i = 0; i < RET.rows(); ++i) {
        LOG(INFO) << "样本:" << i << "　预测类别: " << RET(i, 0) << std::endl;
    }
}

bool knnClassiferTrainer::is_model_trained() {
    return classifer.get_norm_feats().cols() > 0 && classifer.get_norm_feats().rows() > 0;
}