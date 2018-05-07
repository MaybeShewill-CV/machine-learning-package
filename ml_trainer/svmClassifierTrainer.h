/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: svmClassifierTrainer.h
* Date: 18-5-7 下午1:48
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_SVMCLASSIFIERTRAINER_H
#define MACHINE_LEARNING_PACKAGE_SVMCLASSIFIERTRAINER_H

#include <MLTrainerBase.h>
#include <svmClassifier.h>

class svmClassifierTrainer : public MLTrainerBase {
public:
    svmClassifierTrainer() = default;
    ~svmClassifierTrainer() override = default;

    svmClassifierTrainer(int iter_times, double C, double tol, double bias, KERNEL_TYPE kernel_type);

    void train(const std::string& input_file_path) override ;
    void test(const std::string& input_file_path) override ;
    void deploy(const std::string& input_file_path) override ;

private:
    DataLoder _dataLoder;
    svmClassifier _classifier;
    bool is_model_trained() override ;

};


#endif //MACHINE_LEARNING_PACKAGE_SVMCLASSIFIERTRAINER_H
