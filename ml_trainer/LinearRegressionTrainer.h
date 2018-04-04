/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: linearRegressionTrainer.h
* Date: 18-4-4 下午2:48
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_LINEARREGRESSIONTRAINER_H
#define MACHINE_LEARNING_PACKAGE_LINEARREGRESSIONTRAINER_H

#include <MLTrainerBase.h>
#include <LinearRegression.h>

#include <string>

class LinearRegressionTrainer: public MLTrainerBase {
public:
    LinearRegressionTrainer() = default;
    ~LinearRegressionTrainer() override = default;

    void train(const std::string& input_file_path) override;

private:
    DataLoder dataLoder;
    LinearRegression regressor;
};


#endif //MACHINE_LEARNING_PACKAGE_LINEARREGRESSIONTRAINER_H
