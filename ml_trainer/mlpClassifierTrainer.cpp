/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: mlpClassifierTrainer.cpp
* Date: 18-5-2 上午11:21
************************************************/

#include "mlpClassifierTrainer.h"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <file_system_processor.h>

#define DEBUG

mlpClassifierTrainer::mlpClassifierTrainer(int class_nums, double lr,
                                    int epoch_nums, int batch_size,
                                    int input_dims) :
        _class_nums(class_nums), _lr(lr),
        _epoch_nums(epoch_nums), _batch_size(batch_size),
        _input_dims(input_dims){
    _classifier = mlpClassifier(_class_nums, _epoch_nums, _lr, _batch_size, _input_dims);
}

mlpClassifierTrainer::~mlpClassifierTrainer() {}

void mlpClassifierTrainer::train(const std::string &input_file_path) {
    const std::string train_data_file_path = FileSystemProcessor::combine_path(input_file_path,
                                                                               "train-images-idx3-ubyte");
    const std::string train_label_file_path = FileSystemProcessor::combine_path(input_file_path,
                                                                                "train-labels-idx1-ubyte");
    auto mnist_data = _dataloder.load_mnist(train_data_file_path, train_label_file_path);
    LOG(INFO) << "加载" << mnist_data.size() << "条mnist数据集" << std::endl;
    Eigen::MatrixXd X(mnist_data.size(), _input_dims);
    Eigen::MatrixXd Y(mnist_data.size(), 1);
    for (size_t i = 0; i < mnist_data.size(); ++i) {
        for (auto j = 0; j < mnist_data[i].first.rows(); ++j) {
            X(i, j) = mnist_data[i].first(j, 0);
        }
        Y(i, 0) = mnist_data[i].second;
    }
    _classifier.fit(X, Y);
}

void mlpClassifierTrainer::test(const std::string &input_file_path) {
    if (!is_model_trained()) {
        LOG(ERROR) << "模型未训练" << std::endl;
    }

    const std::string test_data_file_path = FileSystemProcessor::combine_path(input_file_path,
                                                                               "t10k-images-idx3-ubyte");
    const std::string test_label_file_path = FileSystemProcessor::combine_path(input_file_path,
                                                                                "t10k-labels-idx1-ubyte");
    auto mnist_data = _dataloder.load_mnist(test_data_file_path, test_label_file_path);
    LOG(INFO) << "加载" << mnist_data.size() << "条mnist数据集" << std::endl;
    Eigen::MatrixXd X(mnist_data.size(), _input_dims);
    Eigen::MatrixXd Y(mnist_data.size(), 1);
    for (size_t i = 0; i < mnist_data.size(); ++i) {
        for (auto j = 0; j < mnist_data[i].first.rows(); ++j) {
            X(i, j) = mnist_data[i].first(j, 0);
        }
        Y(i, 0) = mnist_data[i].second;
    }
    Eigen::MatrixXd preds;
    _classifier.predict(X, preds);

    LOG(INFO) << "测试结果如下" << std::endl;
    LOG(INFO) << "Label: ---- Predict: ----" << std::endl;
    int correct_preds_count = 0;
    for (auto i = 0; i < preds.rows(); ++i) {
        LOG(INFO) << Y(i, 0) << " --- " << preds(i, 0) << std::endl;
        if (Y(i, 0) == preds(i, 0)) {
            correct_preds_count++;
        }
#ifdef DEBUG
        Eigen::MatrixXd image_matrix = X.row(i);
        image_matrix.resize(28, 28);
        image_matrix.transposeInPlace();
        cv::Mat image;
        cv::eigen2cv(image_matrix, image);
        cv::resize(image, image, cv::Size(280, 280));
        cv::imshow("原始minist输入图像", image);
        cv::waitKey(2000);
#endif
    }
    LOG(INFO) << "测试完毕, 准确率为: " << correct_preds_count * 100 / preds.rows() << "%" << std::endl;
}

void mlpClassifierTrainer::deploy(const std::string &input_file_path) {}

bool mlpClassifierTrainer::is_model_trained() {
    return !_classifier.get_mlp_layer().empty();
}
