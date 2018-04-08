/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: knnClassifer.cpp
* Date: 18-4-8 下午6:24
************************************************/

#include "knnClassifer.h"

#include <glog/logging.h>

//#define DEBUG

KnnClassifer::KnnClassifer() {
    distance_type = EUCLIDEAN;
};

KnnClassifer::~KnnClassifer() = default;

KnnClassifer::KnnClassifer(int class_nums, DISTANCE_TYPE distance_type):
        class_nums(class_nums), distance_type(distance_type) {
}

void KnnClassifer::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {

}

void KnnClassifer::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {

}

double KnnClassifer::calculate_distance(const Eigen::ArrayXXd &array_1, const Eigen::ArrayXXd &array_2) {
    double distance = 0.0;
    switch (distance_type) {
        case DISTANCE_TYPE::EUCLIDEAN: {
            Eigen::ArrayXXd diff = array_1 - array_2;
#ifdef DEBUG
            LOG(INFO) << "向量差为: " << diff << std::endl;
#endif
            diff = diff.pow(2);
#ifdef DEBUG
            LOG(INFO) << "向量平方为: " << diff << std::endl;
#endif
            distance = std::sqrt(diff.sum());
#ifdef DEBUG
            LOG(INFO) << "向量间距离为: " << distance << std::endl;
#endif
            break;
        }
        case DISTANCE_TYPE::COSINE: {
            Eigen::ArrayXXd dot_array = array_1 * array_2;
            double dot = dot_array.sum();
#ifdef DEBUG
            LOG(INFO) << "向量点积为: " << dot << std::endl;
#endif
            double array_1_length = std::sqrt(array_1.pow(2).sum());
#ifdef DEBUG
            LOG(INFO) << "向量一模长为: " << array_1_length << std::endl;
#endif
            double array_2_length = std::sqrt(array_2.pow(2).sum());
#ifdef DEBUG
            LOG(INFO) << "向量二模长为: " << array_2_length << std::endl;
#endif
            distance = dot / (array_1_length * array_2_length);
#ifdef DEBUG
            LOG(INFO) << "向量间距离为: " << distance << std::endl;
#endif
            break;
        }
    }
    return distance;
}

