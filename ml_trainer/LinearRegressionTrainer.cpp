/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: linearRegressionTrainer.cpp
* Date: 18-4-4 下午2:48
************************************************/

#include "LinearRegressionTrainer.h"

#include <glog/logging.h>

void LinearRegressionTrainer::train(const std::string &input_file_path) {
    Eigen::MatrixXd input_data;
    dataLoder.load_data_from_txt(input_file_path, input_data);

    Eigen::MatrixXd X(input_data.col(1).rows(), 1);
    Eigen::MatrixXd Y(input_data.col(2).rows(), 1);
    X = input_data.col(1);
    Y = input_data.col(2);

    regressor.fit(X, Y);

    Eigen::MatrixXd preds;
    regressor.predict(X, preds);
    LOG(INFO) << "Label: ---- Predict: ----" << std::endl;
    for (auto i = 0; i < preds.rows(); ++i) {
        LOG(INFO) << X.row(i) << " --- " << Y.row(i) << " --- " << preds.row(i) << std::endl;
    }
}
