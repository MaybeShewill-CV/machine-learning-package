/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: svmClassifierTrainer.cpp
* Date: 18-5-7 下午1:48
************************************************/

#include "svmClassifierTrainer.h"

#include <glog/logging.h>

svmClassifierTrainer::svmClassifierTrainer(const int iter_times, const double C,
                                           const double tol, const double bias,
                                           const KERNEL_TYPE kernel_type) :
        _classifier(iter_times, C, tol, bias, kernel_type) {}

void svmClassifierTrainer::train(const std::string &input_file_path) {
    Eigen::MatrixXd input_data;
    _dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.rows(), input_data.cols() - 1);
    Eigen::MatrixXd Y(input_data.rows(), 1);
    for (auto i  = 0; i < X.cols(); ++i) {
        X.col(i) = input_data.col(i);
    }
    Y = input_data.col(input_data.cols() - 1);

    _classifier.fit(X, Y);
    auto support_vector_nums = 0;
    for (auto i = 0; i < _classifier.get_lagrangian_mul_coffecient().rows(); ++i) {
        if (std::abs(_classifier.get_lagrangian_mul_coffecient()(i)) >= 0.00000001) {
            support_vector_nums += 1;
        }
    }
    LOG(INFO) << "支持向量机训练完毕, 共获取" << support_vector_nums << "个支持向量" << std::endl;
}

void svmClassifierTrainer::test(const std::string &input_file_path) {
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
    _classifier.predict(X, preds);
    LOG(INFO) << "测试结果如下" << std::endl;
    LOG(INFO) << "Label: ---- Predict: ----" << std::endl;
    int correct_preds_count = 0;
    for (auto i = 0; i < preds.rows(); ++i) {
        LOG(INFO) << Y(i, 0) << " --- " << static_cast<int>(preds(i, 0)) << std::endl;
        if (Y(i, 0) == preds(i, 0)) {
            correct_preds_count++;
        }
    }
    LOG(INFO) << "测试完毕, 准确率为: " << correct_preds_count * 100 / preds.rows() << "%" << std::endl;
}

void svmClassifierTrainer::deploy(const std::string &input_file_path) {
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
    _classifier.predict(X, preds);
    LOG(INFO) << "预测结果如下" << std::endl;
    for (auto i = 0; i < preds.rows(); ++i) {
        LOG(INFO) << "样本:" << i + 1 << " 标签: " << static_cast<int>(preds(i, 0)) << std::endl;
    }
}

bool svmClassifierTrainer::is_model_trained() {
    auto coffecient_vec = _classifier.get_lagrangian_mul_coffecient();
    return coffecient_vec.size() > 0;
}