/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: lvqCluster.cpp
* Date: 18-4-19 下午7:32
************************************************/

#include "lvqCluster.h"

#include <map>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include <globalUtils.h>

#define DEBUG

std::default_random_engine E(static_cast<u_long >(time(nullptr)));

namespace lvqCluster_func {
    namespace internal_func {
        std::map<double, cv::Scalar> init_color_map(const int class_nums, const int channel_nums=3) {
            // 设置随机数生成引擎
            std::default_random_engine e;
            std::uniform_int_distribution<unsigned> u(0, 255);

            std::map<double, cv::Scalar> color_map;
            for (auto i = 0; i < class_nums; ++i) {
                if (channel_nums == 3) {
                    cv::Scalar color(u(e), u(e), u(e));
                    color_map.insert(std::make_pair(i, color));
                }
            }
            return color_map;
        };

        void vis_origin_feats_distribute(cv::Mat &image, const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
            auto color_map = init_color_map(2);
            for (auto i = 0; i < X.rows(); ++i) {
                auto pt_x = static_cast<int>(X(i, 0) * 500);
                auto pt_y = 500 - static_cast<int>(X(i, 1) * 500);
                auto label = Y(i, 0);
                cv::Point pt(pt_x, pt_y);
                cv::circle(image, pt, 4, color_map[label], 5, CV_FILLED);
            }
        }

        void draw_cross(cv::Mat &image, cv::Point &cross_center, int line_length, cv::Scalar &color) {
            cv::Point top;
            cv::Point bottom;
            cv::Point left;
            cv::Point right;

            if (cross_center.y - line_length / 2 < 0) {
                top.x = cross_center.x;
                top.y = 0;
            } else {
                top.x = cross_center.x;
                top.y = cross_center.y - line_length / 2;
            }
            if (cross_center.y + line_length / 2 > image.rows) {
                bottom.x = cross_center.x;
                bottom.y = image.rows - 1;
            } else {
                bottom.x = cross_center.x;
                bottom.y = cross_center.y + line_length / 2;
            }
            if (cross_center.x - line_length / 2 < 0) {
                left.x = 0;
                left.y = cross_center.y;
            } else {
                left.x = cross_center.x - line_length / 2;
                left.y = cross_center.y;
            }
            if (cross_center.x + line_length / 2 > image.cols) {
                right.x = image.cols;
                right.y = cross_center.y;
            } else {
                right.x = cross_center.x + line_length / 2;
                right.y = cross_center.y;
            }

            cv::line(image, top, bottom, color, 2);
            cv::line(image, left, right, color, 2);
        }

        void vis_prototype_vec(cv::Mat &image, const Eigen::MatrixXd &X,
                               std::vector<lvq_cluster> &cluster_vec) {
            auto color_map = init_color_map(static_cast<int>(cluster_vec.size()));
            // 画出原型向量的位置
            for (size_t i = 0; i < cluster_vec.size(); ++i) {
                Eigen::RowVectorXd prototype_vec = cluster_vec[i].get_prototype_vec();
                cv::Point pt(static_cast<int>(prototype_vec[0] * 500),
                             500 - static_cast<int>(prototype_vec[1] * 500));
                auto line_length = 40;
                draw_cross(image, pt, line_length, color_map[i]);
            }

            // 画出数据点分布
            for (auto i = 0; i < X.rows(); ++i) {
                Eigen::RowVectorXd feats_vec = X.row(i);
                std::vector<double > dist_vec;
                for (auto &cluster : cluster_vec) {
                    auto prototype_vec = cluster.get_prototype_vec();
                    double distance = std::sqrt((prototype_vec - feats_vec).array().pow(2).sum());
                    dist_vec.push_back(distance);
                }

                std::vector<size_t > sort_idx = GlobalUtils::sort_indexes(dist_vec);
                auto pt_x = static_cast<int>(X(i, 0) * 500);
                auto pt_y = 500 - static_cast<int>(X(i, 1) * 500);
                cv::Point pt(pt_x, pt_y);
                cv::circle(image, pt, 4, color_map[sort_idx[0]], 5, CV_FILLED);
            }
        }
    }
}

lvqCluster::lvqCluster(const int class_nums, const double dist_threshold,
                       DISTANCE_TYPE distance_type, const int loop_times, const double lr) :
        _class_nums(class_nums),  _max_loop_times(loop_times), _dist_threshold(dist_threshold), _lr(lr),
        _distanceType(distance_type) {}

void lvqCluster::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 初始化lvq聚类簇
    init_lvq_cluster(X, Y);

    // 循环迭代训练
    int loop_time = 0;
    while (true) {
        // 根据样本距离重新划分更新聚类簇
        LOG(INFO) << "开始第" << loop_time + 1 << "次迭代更新" << std::endl;
        if (loop_time < _max_loop_times && update_cluster(X, Y)) {
            loop_time++;
        } else {
            LOG(INFO) << "迭代训练完毕" << std::endl;
            cv::Mat origin_sample_distribution(500, 500, CV_8UC3);
            cv::Mat cluster_prototype_distribution(500, 500, CV_8UC3);
            lvqCluster_func::internal_func::vis_origin_feats_distribute(origin_sample_distribution, X, Y);
            lvqCluster_func::internal_func::vis_prototype_vec(cluster_prototype_distribution, X, _cluster_vec);
            cv::imshow("原始数据分布", origin_sample_distribution);
            cv::imshow("lvq聚类原型向量分布", cluster_prototype_distribution);
            cv::waitKey();
            break;
        }
    }
}

void lvqCluster::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::MatrixXd ret(X.rows(), 1);
    for (auto i = 0; i < X.rows(); ++i) {
        ret(i, 0) = predict_one_sample(X.row(i));
    }
    RET = ret;
}

double lvqCluster::predict_one_sample(const Eigen::RowVectorXd &feats) {
    std::vector<double> dist_vec;
    for (auto &lvqcluster : _cluster_vec) {
        dist_vec.push_back(calculate_distance(feats, lvqcluster.get_prototype_vec()));
    }

    std::vector<size_t > sort_idx = GlobalUtils::sort_indexes(dist_vec);

    return _cluster_vec[sort_idx[0]].get_cluster_label();
}


void lvqCluster::init_lvq_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    std::map<double, std::vector<long> > sample_distribute_map;
    for (auto i = 0; i < X.rows(); ++i) {
        auto label = Y(i, 0);
        std::vector<long> index_tmp = {i};
        if (GlobalUtils::has_key(sample_distribute_map, label)) {
            sample_distribute_map[label].push_back(i);
        } else {
            sample_distribute_map.insert(std::make_pair(label, index_tmp));
        }
    }

    for (auto i = 0; i < _class_nums; ++i) {
        std::uniform_int_distribution<unsigned> u(0, static_cast<uint>(X.rows() - 1));
        auto random_index = u(E);
        lvq_cluster cluster_tmp(X.row(random_index), _dist_threshold, _distanceType, Y(random_index, 0), _lr);
        _cluster_vec.push_back(cluster_tmp);
    }
}

bool lvqCluster::update_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 首先随机选取样本
    std::uniform_int_distribution<unsigned> uniform(0, static_cast<uint>(X.rows() - 1));
    auto random_index = uniform(E);
    Eigen::RowVectorXd selected_prototype_vec = X.row(random_index);
    double prototype_vec_label = Y(random_index, 0);

    // 计算与原型向量的距离并选择距离最近的向量进行更新
    std::vector<double> dist_vec;
    for (auto &cluster : _cluster_vec) {
        dist_vec.push_back(calculate_distance(cluster.get_prototype_vec(), selected_prototype_vec));
    }
    std::vector<size_t > sort_idx = GlobalUtils::sort_indexes(dist_vec);

    // 可视化更新过程
#ifdef DEBUG
    cv::Mat origin_image(500, 500, CV_8UC3);
    lvqCluster_func::internal_func::vis_origin_feats_distribute(origin_image, X, Y);
    cv::Mat before_update_image(500, 500, CV_8UC3);
    lvqCluster_func::internal_func::vis_prototype_vec(before_update_image, X, _cluster_vec);
#endif

    // 更新原型向量
    auto &cluster_need_to_update = _cluster_vec[sort_idx[0]];
    LOG(INFO) << "更新前的原型向量" << cluster_need_to_update.get_prototype_vec() << std::endl;
    Eigen::RowVectorXd diff_prototype_vec = selected_prototype_vec - cluster_need_to_update.get_prototype_vec();
    LOG(INFO) << "更新差分原型向量" << diff_prototype_vec << std::endl;
    bool is_similar(prototype_vec_label == cluster_need_to_update.get_cluster_label());

     if (cluster_need_to_update.is_cluster_updated(diff_prototype_vec, prototype_vec_label, is_similar)) {
         LOG(INFO) << "更新后的原型向量" << _cluster_vec[sort_idx[0]].get_prototype_vec() << std::endl;
#ifdef DEBUG
         cv::Mat after_update_image(500, 500, CV_8UC3);
         lvqCluster_func::internal_func::vis_prototype_vec(after_update_image, X, _cluster_vec);
         cv::imshow("原始数据分布图", origin_image);
         cv::imshow("更新前的数据分布", before_update_image);
         cv::imshow("更新后的数据分布", after_update_image);
         cv::waitKey(250);
#endif
         return true;
     } else {
         return false;
     }
}

double lvqCluster::calculate_distance(const Eigen::RowVectorXd &vectorXd_1,
                                         const Eigen::RowVectorXd &vectorXd_2) {
    double distance = 0.0;
    Eigen::ArrayXXd array_1 = vectorXd_1.array();
    Eigen::ArrayXXd array_2 = vectorXd_2.array();
    switch (_distanceType) {
        case DISTANCE_TYPE::EUCLIDEAN: {
            Eigen::ArrayXXd diff = array_1 - array_2;
#ifdef DISDEBUG
            LOG(INFO) << "向量差为: " << diff << std::endl;
#endif
            diff = diff.pow(2);
#ifdef DISDEBUG
            LOG(INFO) << "向量平方为: " << diff << std::endl;
#endif
            distance = std::sqrt(diff.sum());
#ifdef DISDEBUG
            LOG(INFO) << "向量间距离为: " << distance << std::endl;
#endif
            break;
        }
        case DISTANCE_TYPE::COSINE: {
            Eigen::ArrayXXd dot_array = array_1 * array_2;
            double dot = dot_array.sum();
#ifdef DISDEBUG
            LOG(INFO) << "向量点积为: " << dot << std::endl;
#endif
            double array_1_length = std::sqrt(array_1.pow(2).sum());
#ifdef DISDEBUG
            LOG(INFO) << "向量一模长为: " << array_1_length << std::endl;
#endif
            double array_2_length = std::sqrt(array_2.pow(2).sum());
#ifdef DISDEBUG
            LOG(INFO) << "向量二模长为: " << array_2_length << std::endl;
#endif
            distance = dot / (array_1_length * array_2_length);
#ifdef DISDEBUG
            LOG(INFO) << "向量间距离为: " << distance << std::endl;
#endif
            break;
        }
    }
    return distance;
}

void lvqCluster::test() {
    Eigen::MatrixXd X(30, 2);
    Eigen::MatrixXd Y(30, 1);
    X << 0.697, 0.460, 0.774, 0.376, 0.634, 0.264, 0.608, 0.318, 0.556, 0.215, 0.403, 0.237, 0.481, 0.149, 0.437,
            0.211, 0.666, 0.091, 0.243, 0.267, 0.245, 0.057, 0.343, 0.099, 0.639, 0.161, 0.657, 0.198, 0.360, 0.370,
            0.593, 0.042, 0.719, 0.103, 0.359, 0.188, 0.339, 0.241, 0.282, 0.257, 0.748, 0.232, 0.714, 0.346, 0.483,
            0.312, 0.478, 0.437, 0.525, 0.369, 0.751, 0.489, 0.532, 0.472, 0.473, 0.376, 0.725, 0.445, 0.446, 0.459;
    Y << 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    fit(X, Y);

    Eigen::MatrixXd ret;
    predict(X, ret);
    LOG(INFO) << "预测结果如下: " << std::endl;
    LOG(INFO) << ret << std::endl;
    auto diff = Y - ret;
    auto correct_prediction = (diff.array() == 0).count();
    LOG(INFO) << "预测准确率: " << static_cast<double>(correct_prediction) / Y.rows() << std::endl;
}