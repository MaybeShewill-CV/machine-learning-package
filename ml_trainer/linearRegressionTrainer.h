/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: linearRegressionTrainer.h
* Date: 18-4-4 下午2:48
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_LINEARREGRESSIONTRAINER_H
#define MACHINE_LEARNING_PACKAGE_LINEARREGRESSIONTRAINER_H

#include <MLTrainerBase.h>
#include <linearRegression.h>

#include <string>

class LinearRegressionTrainer: public MLTrainerBase {
public:
    LinearRegressionTrainer() = default;
    ~LinearRegressionTrainer() override = default;

    LinearRegressionTrainer(double lr, int iter_times);

    void train(const std::string& input_file_path) override;
    void test(const std::string& input_file_path) override;
    void deploy(const std::string& input_file_path) override;

private:
    DataLoder dataLoder;
    LinearRegression regressor;

    bool is_model_trained() override;
};


#endif //MACHINE_LEARNING_PACKAGE_LINEARREGRESSIONTRAINER_H
