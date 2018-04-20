/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: lvqClusterTrainer.h
* Date: 18-4-20 下午5:34
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_LVQCLUSTERTRAINER_H
#define MACHINE_LEARNING_PACKAGE_LVQCLUSTERTRAINER_H

#include <MLTrainerBase.h>
#include <lvqCluster.h>

class lvqClusterTrainer : MLTrainerBase {
public:
    lvqClusterTrainer() = default;
    ~lvqClusterTrainer() override = default;

    lvqClusterTrainer(int class_nums, int max_iter_times, double lr,
                      DISTANCE_TYPE distance_type = EUCLIDEAN,
                      double dist_threshold = 0.0);

    void train(const std::string& input_file_path) override ;
    void test(const std::string& input_file_path) override ;
    void deploy(const std::string& input_file_path) override ;

private:
    DataLoder dataLoder;
    int _class_nums = 2;
    int _max_iter_times = 1000;
    double _lr = 0.1;
    DISTANCE_TYPE _distanceType = EUCLIDEAN;
    double _dist_threshold = 0.0;
    lvqCluster _cluster = lvqCluster(_class_nums, _dist_threshold,
                                    _distanceType, _max_iter_times, _lr);

    bool is_model_trained() override ;

};


#endif //MACHINE_LEARNING_PACKAGE_LVQCLUSTERTRAINER_H
