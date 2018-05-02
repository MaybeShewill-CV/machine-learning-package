/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: mlpClassifierTrainer.h
* Date: 18-5-2 上午11:21
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_MLPCLASSIFIERTRAINER_H
#define MACHINE_LEARNING_PACKAGE_MLPCLASSIFIERTRAINER_H

#include <MLTrainerBase.h>
#include <mlpClassifier.h>
#include <mnistDataloder.h>

class mlpClassifierTrainer : public MLTrainerBase {
public:
    mlpClassifierTrainer() = default;
    ~mlpClassifierTrainer() override ;

    mlpClassifierTrainer(int class_nums, double lr, int epoch_nums, int batch_size, int input_dims);

    void train(const std::string& input_file_path) override ;
    void test(const std::string& input_file_path) override ;
    void deploy(const std::string& input_file_path) override ;

private:
    int _class_nums = 0;
    double _lr = 0.0f;
    int _epoch_nums = 0;
    int _batch_size = 0;
    int _input_dims = 0;

    mnist_dataloder _dataloder;
    mlpClassifier _classifier;
    bool is_model_trained() override ;
};

#endif //MACHINE_LEARNING_PACKAGE_MLPCLASSIFIERTRAINER_H
