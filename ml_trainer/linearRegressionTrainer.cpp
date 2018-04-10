/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: linearRegressionTrainer.cpp
* Date: 18-4-4 下午2:48
************************************************/

#include "linearRegressionTrainer.h"

#include <algorithm>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <globalUtils.h>

LinearRegressionTrainer::LinearRegressionTrainer(double lr, int iter_times):
        regressor(lr, iter_times) {}

void LinearRegressionTrainer::train(const std::string &input_file_path) {
    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.col(1).rows(), 1);
    Eigen::MatrixXd Y(input_data.col(2).rows(), 1);
    X = input_data.col(1);
    Y = input_data.col(2);

    regressor.fit(X, Y);
}

void LinearRegressionTrainer::test(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(ERROR) << "模型还未训练" << std::endl;
        return;
    }

    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.col(1).rows(), 1);
    Eigen::MatrixXd Y(input_data.col(2).rows(), 1);
    X = input_data.col(1);
    Y = input_data.col(2);

    Eigen::MatrixXd preds;
    regressor.predict(X, preds);
    LOG(INFO) << "测试结果如下" << std::endl;
    LOG(INFO) << "Label: ---- Predict: ----" << std::endl;
    for (auto i = 0; i < preds.rows(); ++i) {
        LOG(INFO) << X.row(i) << " --- " << Y.row(i) << " --- " << preds.row(i) << std::endl;
    }

    // 画出测试示意图
    const int figure_rows = 360;
    const int figure_cols = 480;
    const int margin = 20;

    cv::Mat figure(figure_rows + margin, figure_cols + margin, CV_8UC3);
    std::vector<double> X_vec;
    std::vector<double> Y_vec;
    std::vector<double> Preds_vec;

    for (auto i = 0; i < X.rows(); ++i) {
        X_vec.push_back(X(i, 0));
        Y_vec.push_back(Y(i, 0));
        Preds_vec.push_back(preds(i, 0));
    }

    std::vector<double> X_Scale;
    std::vector<double> Y_Scale;
    std::vector<double> Preds_Scale;

    std::vector<size_t > sort_index = GlobalUtils::sort_indexes(X_vec);
    for (auto i : sort_index) {
        X_Scale.push_back(X_vec[i]);
        Y_Scale.push_back(Y_vec[i]);
        Preds_Scale.push_back(Preds_vec[i]);
    }

    auto X_max = *std::max_element(X_Scale.begin(), X_Scale.end());
    auto Y_max = *std::max_element(Y_Scale.begin(), Y_Scale.end());
    auto Pred_max = *std::max_element(Preds_Scale.begin(), Preds_Scale.end());

    for (size_t i = 0; i < X_Scale.size(); ++i) {
        X_Scale[i] = X_Scale[i] * figure_cols / X_max;
        Y_Scale[i] = Y_Scale[i] * figure_rows / Y_max;
        Preds_Scale[i] = Preds_Scale[i] * figure_rows / Pred_max;
    }

    const cv::Scalar label_color(0, 255, 0);
    const cv::Scalar preds_color(0, 0, 255);

    for (size_t i = 0; i < X_Scale.size(); ++i) {
        cv::Point2d pt(static_cast<int>(X_Scale[i]), static_cast<int>(Y_Scale[i]));
        cv::circle(figure, pt, 1, label_color, 1, CV_FILLED);
    }

    for (size_t i = 0; i < X_Scale.size() - 1; ++i) {
        cv::Point2d pt1(static_cast<int>(X_Scale[i]), static_cast<int>(Preds_Scale[i]));
        cv::Point2d pt2(static_cast<int>(X_Scale[i + 1]), static_cast<int>(Preds_Scale[i + 1]));
        cv::line(figure, pt1, pt2, preds_color);
    }

    cv::imshow("Test Result", figure);
    cv::waitKey();
}

void LinearRegressionTrainer::deploy(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(ERROR) << "模型还未训练" << std::endl;
        return;
    }

    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd test_input(input_data.col(1).rows(), 1);
    test_input = input_data.col(1);
    Eigen::MatrixXd preds;
    regressor.predict(test_input, preds);

    LOG(INFO) << "Input: ---- Predict: ----" << std::endl;
    for (auto i = 0; i < preds.rows(); ++i) {
        LOG(INFO) << test_input.row(i) << " --- " << preds.row(i) << std::endl;
    }
}

bool LinearRegressionTrainer::is_model_trained() {
    return regressor.get_kernel().cols() > 0 && regressor.get_kernel().rows() > 0;
}
