/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: knnClassiferTrainer.h
* Date: 18-4-10 上午10:18
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_KNNCLASSIFERTRAINER_H
#define MACHINE_LEARNING_PACKAGE_KNNCLASSIFERTRAINER_H

#include <MLTrainerBase.h>
#include <knnClassifer.h>

class knnClassiferTrainer: public MLTrainerBase {
public:
    knnClassiferTrainer() = default;
    ~knnClassiferTrainer() override = default;

    void train(const std::string& input_file_path) override;
    void test(const std::string& input_file_path) override;
    void deploy(const std::string& input_file_path) override;

private:
    DataLoder dataLoder;
    KnnClassifer classifer;

    bool is_model_trained() override;
};


#endif //MACHINE_LEARNING_PACKAGE_KNNCLASSIFERTRAINER_H
