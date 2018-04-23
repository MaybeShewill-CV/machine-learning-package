/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: gmmClusterTrainer.h
* Date: 18-4-23 下午3:29
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_GMMCLUSTERTRAINER_H
#define MACHINE_LEARNING_PACKAGE_GMMCLUSTERTRAINER_H

#include <MLTrainerBase.h>
#include <gmmCluster.h>

class gmmClusterTrainer : MLTrainerBase {
public:
    gmmClusterTrainer();
    ~gmmClusterTrainer() override = default;

    gmmClusterTrainer(int class_nums, int max_iter_times, DISTANCE_TYPE distanceType = EUCLIDEAN);

    void train(const std::string& input_file_path);
    void test(const std::string& input_file_path);
    void deploy(const std::string& input_file_path);

private:
    int _class_nums = 2;
    int _max_iter_times = 1000;
    DISTANCE_TYPE _distanceType = EUCLIDEAN;

    DataLoder _dataLoder;
    gmmCluster _cluster;

    bool is_model_trained() override ;

};


#endif //MACHINE_LEARNING_PACKAGE_GMMCLUSTERTRAINER_H
