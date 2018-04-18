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

    explicit decisionTreeClassiferTrainer(DTREE_TYPE dtree_type);

    void train(const std::string& input_file_path) override ;
    void test(const std::string& input_file_path) override ;
    void deploy(const std::string& input_file_path) override ;

private:
    DTREE_TYPE dtreeType = ID3_DTREE;
    DataLoder dataLoder;
    decisionTree classifier = decisionTree(dtreeType);
    bool is_model_trained() override ;
};

#endif //MACHINE_LEARNING_PACKAGE_DECISIONTREECLASSIFERTRAINER_H
