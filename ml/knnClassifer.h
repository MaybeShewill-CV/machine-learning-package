/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: knnClassifer.h
* Date: 18-4-8 下午6:24
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_KNNCLASSIFER_H
#define MACHINE_LEARNING_PACKAGE_KNNCLASSIFER_H

#include <MLBase.h>
#include <glog/logging.h>

#include <eigen3/Eigen/Dense>

class KnnClassifer: public MLBase {
public:
    KnnClassifer();
    ~KnnClassifer() override;

    enum DISTANCE_TYPE {
        EUCLIDEAN,
        COSINE
    };

    KnnClassifer(int class_nums, DISTANCE_TYPE distance_type);

    void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) override;
    void predict(const Eigen::MatrixXd& X, Eigen::MatrixXd& RET) override;

    void test_distance(const Eigen::ArrayXXd &array_1, const Eigen::ArrayXXd &array_2) {
        double distance = calculate_distance(array_1, array_2);
        LOG(INFO) << "计算距离是: " << distance << std::endl;
    }

private:
    const int class_nums = 0;
    DISTANCE_TYPE distance_type;
    double calculate_distance(const Eigen::ArrayXXd &array_1, const Eigen::ArrayXXd &array_2);
};


#endif //MACHINE_LEARNING_PACKAGE_KNNCLASSIFER_H
