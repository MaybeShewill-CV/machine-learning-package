/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: kmeansClusterTrainer.h
* Date: 18-4-19 下午4:53
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_KMEANSCLUSTERTRAINER_H
#define MACHINE_LEARNING_PACKAGE_KMEANSCLUSTERTRAINER_H

#include <MLTrainerBase.h>

#include <kmeansCluster.h>

class kmeansClusterTrainer : public MLTrainerBase {
public:
    kmeansClusterTrainer() = default;
    ~kmeansClusterTrainer() override = default;

    kmeansClusterTrainer(int class_nums, int max_iter_times, DISTANCE_TYPE distance_type);

    void train(const std::string& input_file_path);
    void test(const std::string& input_file_path);
    void deploy(const std::string& input_file_path);

private:
    DataLoder dataLoder;
    bool is_model_trained() override ;

    kmeansCluster _cluster;

    const int _class_nums = 2;
    const int _max_iter_times = 1000;
    DISTANCE_TYPE _distanceType = EUCLIDEAN;
};


#endif //MACHINE_LEARNING_PACKAGE_KMEANSCLUSTERTRAINER_H
