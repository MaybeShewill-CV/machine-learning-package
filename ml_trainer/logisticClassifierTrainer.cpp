/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: logisticClassifierTrainer.cpp
* Date: 18-4-10 下午5:33
************************************************/

#include "logisticClassifierTrainer.h"

#include <glog/logging.h>

logisticClassifierTrainer::logisticClassifierTrainer(double lr, int iter_times):
        classifier(lr, iter_times) {}

void logisticClassifierTrainer::train(const std::string &input_file_path) {
    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.col(1).rows(), input_data.cols() - 2);
    Eigen::MatrixXd Y(input_data.col(2).rows(), 1);
    for (auto i  = 1; i < X.cols(); ++i) {
        X.col(i - 1) = input_data.col(i);
    }
    Y = input_data.col(2);

    classifier.fit(X, Y);
}

void logisticClassifierTrainer::test(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(INFO) << "模型未训练" << std::endl;
        return;
    }

    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.col(1).rows(), input_data.cols() - 2);
    Eigen::MatrixXd Y(input_data.col(2).rows(), 1);
    for (auto i  = 1; i < X.cols(); ++i) {
        X.col(i - 1) = input_data.col(i);
    }
    Y = input_data.col(2);

    Eigen::MatrixXd preds;
    classifier.predict(X, preds);
    LOG(INFO) << "测试结果如下" << std::endl;
    LOG(INFO) << "Label: ---- Predict: ----" << std::endl;
    int correct_preds_count = 0;
    for (auto i = 0; i < preds.rows(); ++i) {
        LOG(INFO) << Y(i, 0) << " --- " << preds(i, 0) << std::endl;
        if (Y(i, 0) == preds(i, 0)) {
            correct_preds_count++;
        }
    }
    LOG(INFO) << "测试完毕, 准确率为: " << correct_preds_count * 100 / preds.rows() << "%" << std::endl;
}

void logisticClassifierTrainer::deploy(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(INFO) << "模型未训练" << std::endl;
        return;
    }
}

bool logisticClassifierTrainer::is_model_trained() {
    return classifier.get_kernel().cols() > 0 && classifier.get_kernel().rows() > 0;
}