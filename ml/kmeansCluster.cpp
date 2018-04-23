/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: kmeansCluster.cpp
* Date: 18-4-18 下午3:26
************************************************/

#include "kmeansCluster.h"

#include <ctime>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <globalUtils.h>

#define DEBUG

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

    void vis_feats(cv::Mat &image, const Eigen::MatrixXd &sample_feats_matrix, const cv::Scalar &color) {
        Eigen::RowVectorXd mean_vec = sample_feats_matrix.colwise().mean();
        cv::Point center(static_cast<int>(mean_vec(0, 0) * 500), static_cast<int>(mean_vec(0, 1) * 500));
        draw_cross(image, center, 40, color);
        for (auto i = 0; i < sample_feats_matrix.rows(); ++i) {
            auto pt_x = static_cast<int>(sample_feats_matrix(i, 0) * 500);
            auto pt_y = 500 - static_cast<int>(sample_feats_matrix(i, 1) * 500);
            cv::Point pt(pt_x, pt_y);
            cv::circle(image, pt, 4, color, 5, CV_FILLED);
        }
    }
}

kmeansCluster::kmeansCluster() {
    // 初始化聚类簇
    for (auto i = 0; i < _class_nums; ++i) {
        _cluster_vec.emplace_back(cluster());
    }
}

kmeansCluster::kmeansCluster(const int class_nums, const double dist_threshold,
                             DISTANCE_TYPE distance_type, const int loop_times) :
        _class_nums(class_nums),_max_loop_times(loop_times), _dist_threshold(dist_threshold),
        _distanceType(distance_type) {
    // 初始化聚类簇
    for (auto i = 0; i < _class_nums; ++i) {
        _cluster_vec.emplace_back(cluster());
    }
}

void kmeansCluster::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 随机初始化聚类簇
    init_kmeans_cluster(X, Y);

    // 计算距离矩阵
    auto loop_time = 0;
    while (true) {
        Eigen::MatrixXd dist_matrix = calculate_distance_matrix(X);
        // 根据样本距离重新划分更新聚类簇
        LOG(INFO) << "开始第" << loop_time + 1 << "次迭代更新" << std::endl;
        if (loop_time <= _max_loop_times && update_cluster(X, Y, dist_matrix)) {
            loop_time++;
        } else {
            LOG(INFO) << "迭代训练完毕" << std::endl;
            break;
        }
    }
}

void kmeansCluster::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::MatrixXd ret(X.rows(), 1);
    for (auto i = 0; i < X.rows(); ++i) {
        ret(i, 0) = predict_one_sample(X.row(i));
    }
    RET = ret;
}

double kmeansCluster::predict_one_sample(const Eigen::RowVectorXd &feats) {
    std::vector<double> dist_vec;
    for (auto &cluster : _cluster_vec) {
        dist_vec.push_back(calculate_distance(feats, cluster.get_cluster_mean_feats_vec()));
    }
    std::vector<size_t > sort_idx = GlobalUtils::sort_indexes(dist_vec);

    return _cluster_vec[sort_idx[0]].get_cluster_label();
}

void kmeansCluster::init_kmeans_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 设置随机数生成引擎
    std::default_random_engine e(static_cast<u_long >(time(0)));
    std::uniform_int_distribution<unsigned> u(0, static_cast<unsigned>(X.rows() - 1));

    std::vector<uint> random_index_vec;
    LOG(INFO) << "开始初始化" << _class_nums << "个聚类簇" << std::endl;
    for (auto i = 0; i < _class_nums; ++i) {
        uint random_index = u(e);
        while (GlobalUtils::has_elements(random_index_vec, random_index)) {
            random_index = u(e);
        }
#ifdef DEBUG
        LOG(INFO) << "--簇" << i << "初始化向量为: " << X.row(random_index)
                  << " 初始化标签为: " << Y.row(random_index) << std::endl;
#endif
        _cluster_vec[i] = cluster(X.row(random_index), Y.row(random_index));
    }
}

Eigen::MatrixXd kmeansCluster::calculate_distance_matrix(const Eigen::MatrixXd &X) {
    Eigen::MatrixXd distance_matrix(X.rows(), _class_nums);
    for (auto i = 0; i < X.rows(); ++i) {
        for (auto j = 0; j < _class_nums; ++j) {
            distance_matrix(i, j) = calculate_distance(X.row(i), _cluster_vec[j].get_cluster_mean_feats_vec());
        }
    }

    return distance_matrix;
}

bool kmeansCluster::update_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
                                   const Eigen::MatrixXd &dist_matrix) {
    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd> > cluster_matrix_pair(
            static_cast<uint >(_class_nums), std::make_pair(Eigen::MatrixXd(), Eigen::MatrixXd()));
    std::vector<std::vector<long> > split_index_vec(static_cast<uint>(_class_nums), std::vector<long>());
    for (auto i = 0; i < dist_matrix.rows(); ++i) {
        Eigen::MatrixXd::Index minRow, minCol;
        dist_matrix.row(i).minCoeff(&minRow, &minCol);
        split_index_vec[static_cast<uint>(minCol)].push_back(i);
    }

    for (size_t i = 0; i < split_index_vec.size(); ++i) {
        Eigen::MatrixXd _x_tmp(split_index_vec[i].size(), X.cols());
        Eigen::MatrixXd _y_tmp(split_index_vec[i].size(), Y.cols());
        for (size_t j = 0; j < split_index_vec[i].size(); ++j) {
            _x_tmp.row(j) = X.row(split_index_vec[i][j]);
            _y_tmp.row(j) = Y.row(split_index_vec[i][j]);
        }
        cluster_matrix_pair[i] = std::make_pair(_x_tmp, _y_tmp);
    }

    // 聚类效果可视化
#ifdef DEBUG
    auto color_map = internal_func::init_color_map(_class_nums);
    cv::Mat before_cluster_image(500, 500, CV_8UC3);
    for (auto i = 0; i < _class_nums; ++i) {
        internal_func::vis_feats(before_cluster_image, _cluster_vec[i].get_sample_feats_matrix(), color_map[i]);
    }
#endif

    bool is_cluster_updated = false;
    for (size_t i = 0; i < cluster_matrix_pair.size(); ++i) {
        auto sample_feats_matrix = cluster_matrix_pair[i].first;
        auto label_matrix = cluster_matrix_pair[i].second;
        auto origin_mean_feats = _cluster_vec[i].get_cluster_mean_feats_vec();
        if (_cluster_vec[i].is_cluster_updated(sample_feats_matrix, label_matrix)) {
            is_cluster_updated = true;
        } else {
            LOG(INFO) << "--聚类簇" << i << "均值向量未更新" << std::endl;
        }
    }

#ifdef DEBUG
    cv::Mat after_cluster_image(500, 500, CV_8UC3);
    for (auto i = 0; i < _class_nums; ++i) {
        internal_func::vis_feats(after_cluster_image, _cluster_vec[i].get_sample_feats_matrix(), color_map[i]);
    }

    cv::Mat origin_data_distribute(500, 500, CV_8UC3);
    internal_func::vis_origin_feats_distribute(origin_data_distribute, X, Y);

    cv::imshow("origin data", origin_data_distribute);
    cv::imshow("before cluster", before_cluster_image);
    cv::imshow("after cluster", after_cluster_image);
    cv::waitKey(2000);
#endif

    return is_cluster_updated;
}

double kmeansCluster::calculate_distance(const Eigen::RowVectorXd &vectorXd_1,
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

void kmeansCluster::test() {
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