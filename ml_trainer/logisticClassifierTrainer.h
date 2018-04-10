/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: logisticClassifierTrainer.h
* Date: 18-4-10 下午5:33
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_LOGISTICCLASSIFIERTRAINER_H
#define MACHINE_LEARNING_PACKAGE_LOGISTICCLASSIFIERTRAINER_H

#include <MLTrainerBase.h>

#include <logisticRegression.h>

class logisticClassifierTrainer: MLTrainerBase {
public:
    logisticClassifierTrainer() = default;
    ~logisticClassifierTrainer() override = default;

    logisticClassifierTrainer(double lr, int iter_times);

    void train(const std::string& input_file_path) override ;
    void test(const std::string& input_file_path) override ;
    void deploy(const std::string& input_file_path) override ;

private:
    DataLoder dataLoder;
    logisticRegression classifier;

    bool is_model_trained() override ;

};


#endif //MACHINE_LEARNING_PACKAGE_LOGISTICCLASSIFIERTRAINER_H
