/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: dbscanClusterTrainer.h
* Date: 18-4-24 下午2:19
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_DBSCANCLUSTERTRAINER_H
#define MACHINE_LEARNING_PACKAGE_DBSCANCLUSTERTRAINER_H

#include <MLTrainerBase.h>
#include <dbscanCluster.h>

class dbscanClusterTrainer : public MLTrainerBase{
public:
    dbscanClusterTrainer() = default;
    ~dbscanClusterTrainer() override = default;

    dbscanClusterTrainer(int core_obj_nums_threshold, double dist_threshold,
                         DISTANCE_TYPE distancetype = EUCLIDEAN);

    void train(const std::string& input_file_path) override ;
    void test(const std::string& input_file_path) override ;
    void deploy(const std::string& input_file_path) override ;

private:
    DataLoder _dataLoder;
    int __core_obj_nums_threshold = 0;
    double _dist_threshold = 0.0;
    DISTANCE_TYPE _distanceType = EUCLIDEAN;
    dbscanCluster _cluster;

    bool is_model_trained() override ;
};


#endif //MACHINE_LEARNING_PACKAGE_DBSCANCLUSTERTRAINER_H
