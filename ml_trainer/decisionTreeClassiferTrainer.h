/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: decisionTreeClassiferTrainer.h
* Date: 18-4-16 下午7:31
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_DECISIONTREECLASSIFERTRAINER_H
#define MACHINE_LEARNING_PACKAGE_DECISIONTREECLASSIFERTRAINER_H

#include <MLTrainerBase.h>

#include <decisionTree.h>

class decisionTreeClassiferTrainer : public MLTrainerBase {
public:
    decisionTreeClassiferTrainer() = default;
    ~decisionTreeClassiferTrainer() override = default;

    void train(const std::string& input_file_path) override ;
    void test(const std::string& input_file_path) override ;
    void deploy(const std::string& input_file_path) override ;

private:
    DataLoder dataLoder;
    decisionTree classifier;
    bool is_model_trained() override ;
};


#endif //MACHINE_LEARNING_PACKAGE_DECISIONTREECLASSIFERTRAINER_H
