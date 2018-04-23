/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: gmmCluster.cpp
* Date: 18-4-20 下午7:41
************************************************/

#include "gmmCluster.h"

#include <map>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <globalUtils.h>

#define DEBUG

namespace gmmCluster_internal {
    std::default_random_engine E(static_cast<u_long >(time(nullptr)));

    template <typename T>
    void print_vec(const std::vector<T> vec) {
        for (size_t i = 0; i < vec.size(); ++i) {
            LOG(INFO) << "元素:" << i << " 值为: " << vec[i] << std::endl;
        }
    }

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
                           std::vector<gmm_cluster> &cluster_vec) {
        auto color_map = init_color_map(static_cast<int>(cluster_vec.size()));
        // 画出原型向量的位置
        for (size_t i = 0; i < cluster_vec.size(); ++i) {
            Eigen::RowVectorXd prototype_vec = cluster_vec[i].get_mean_vec();
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
                auto prototype_vec = cluster.get_mean_vec();
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

gmmCluster::gmmCluster(const int class_nums, const int max_iter_times, DISTANCE_TYPE distance_type) :
        _class_nums(class_nums), _max_item_times(max_iter_times), _distanceType(distance_type){

}

void gmmCluster::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 初始化高斯混合模型
    init_gmm_cluster(X, Y);

    // EM算法迭代更新参数
    auto loop_time = 0;
    while (true) {
        // 根据样本距离重新划分更新聚类簇
        LOG(INFO) << "开始第" << loop_time + 1 << "次迭代更新" << std::endl;
        if (loop_time <= _max_item_times && update_cluster(X, Y, loop_time + 1)) {
            loop_time++;
        } else {
            LOG(INFO) << "迭代训练完毕" << std::endl;
            break;
        }
    }

    // 训练完毕后为每个聚类簇赋标签
    vote_cluster_label(X, Y);
}

void gmmCluster::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::MatrixXd ret(X.rows(), 1);
    for (auto i = 0; i < X.rows(); ++i) {
        ret(i, 0) = predict_one_smaple(X.row(i));
    }
    RET = ret;
}

double gmmCluster::predict_one_smaple(const Eigen::RowVectorXd &input) {
    std::vector<double> dist_vec;
    for (auto &cluster : _cluster_vec) {
        dist_vec.push_back(calculate_distance(input, cluster.get_mean_vec()));
    }
    std::vector<size_t > sort_idx = GlobalUtils::sort_indexes(dist_vec);
    return _cluster_vec[sort_idx[0]].get_cluster_label();
}

void gmmCluster::init_gmm_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 初始化簇向量
    // TODO 检查高斯混合模型初始化过程是否正确
//    std::vector<long> random_index_vec = {5, 21, 26};
    for (auto i = 0; i < _class_nums; ++i) {
        std::uniform_int_distribution<long > u(0, X.rows() - 1);
//        auto random_index = random_index_vec[i];
        auto random_index = u(gmmCluster_internal::E);
        gmm_cluster cluster_tmp(X.row(random_index), 1.0 / _class_nums);
        _cluster_vec.push_back(cluster_tmp);
    }
}

Eigen::MatrixXd gmmCluster::compute_gmm_posterior_prob_matrix(const Eigen::MatrixXd &X) {
    Eigen::MatrixXd posterior_prob_matrix(X.rows(), _class_nums);
    for (auto i = 0; i < X.rows(); ++i) {
        Eigen::RowVectorXd feats_vec = X.row(i);
        std::vector<double> prob_density_vec;
        for (auto &cluster : _cluster_vec) {
            prob_density_vec.push_back(cluster.get_gmm_mix_coefficient() *
                                               cluster.compute_prob_density(feats_vec));
        }

        double prob_density_sum = 0.0;
        for (auto &prob_density : prob_density_vec) {
            prob_density_sum += prob_density;
        }

        for (auto j = 0; j < _class_nums; ++j) {
            posterior_prob_matrix(i, j) = prob_density_vec[j] / prob_density_sum;
        }
    }
    return posterior_prob_matrix;
}

bool gmmCluster::update_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const int iter_time) {
    Eigen::MatrixXd posterior_prob_matrix = compute_gmm_posterior_prob_matrix(X);
#ifdef DEBUG
    LOG(INFO) << "高斯混合模型后验概率矩阵为: " << posterior_prob_matrix << std::endl;
#endif
    Eigen::RowVectorXd posterior_prob_sum = posterior_prob_matrix.colwise().sum();

    // 求解更新后的均值向量
    std::vector<Eigen::RowVectorXd> update_mean_vec;
    for (auto i = 0; i < _class_nums; ++i) {
        Eigen::RowVectorXd tmp = posterior_prob_matrix.col(i).transpose() * X;
        update_mean_vec.emplace_back(tmp / posterior_prob_sum(0, i));
    }
    // 求解更新后的协方差矩阵
    std::vector<Eigen::MatrixXd> update_covariance_matrix_vec;
    for (auto i = 0; i < _class_nums; ++i) {
        Eigen::MatrixXd covariance_tmp = Eigen::MatrixXd::Zero(X.cols(), X.cols());
        for (auto j = 0; j < X.rows(); ++j) {
            covariance_tmp += posterior_prob_matrix(j, i) *
                    ((X.row(j) - update_mean_vec[i]).transpose() * (X.row(j) - update_mean_vec[i]));
        }
        covariance_tmp = covariance_tmp / posterior_prob_sum(0, i);
        update_covariance_matrix_vec.push_back(covariance_tmp);
    }
    // 求解更新后的高斯混合模型参数
    std::vector<double> update_gmm_mix_coefficient_vec;
    for (auto i = 0; i < _class_nums; ++i) {
        update_gmm_mix_coefficient_vec.push_back(posterior_prob_sum(0, i) / X.rows());
    }

    LOG(INFO) << "第" << iter_time << "次更新后的均值向量如下:" << std::endl;
    gmmCluster_internal::print_vec(update_mean_vec);
    LOG(INFO) << "第" << iter_time << "次更新后的协方差矩阵如下:" << std::endl;
    gmmCluster_internal::print_vec(update_covariance_matrix_vec);
    LOG(INFO) << "第" << iter_time << "次更新后的高斯混合模型参数如下:" << std::endl;
    gmmCluster_internal::print_vec(update_gmm_mix_coefficient_vec);

#ifdef DEBUG
    // 可视化数据更新过程
    cv::Mat origin_data_distribution(500, 500, CV_8UC3);
    gmmCluster_internal::vis_origin_feats_distribute(origin_data_distribution, X, Y);
    cv::Mat before_update_image(500, 500, CV_8UC3);
    gmmCluster_internal::vis_prototype_vec(before_update_image, X, _cluster_vec);
#endif

    bool update_flag = false;
    for (size_t i = 0; i < _cluster_vec.size(); ++i) {
        auto &cluster = _cluster_vec[i];
        if (cluster.update_gmm_cluster(update_mean_vec[i], update_covariance_matrix_vec[i],
                                       update_gmm_mix_coefficient_vec[i])) {
            update_flag = true;
        }
    }

#ifdef DEBUG
    cv::Mat after_update_image(500, 500, CV_8UC3);
    gmmCluster_internal::vis_prototype_vec(after_update_image, X, _cluster_vec);
    cv::imshow("原始数据分布图", origin_data_distribution);
    cv::imshow("更新前的数据分布", before_update_image);
    cv::imshow("更新后的数据分布", after_update_image);
    cv::waitKey(250);
#endif
    return update_flag;
}

void gmmCluster::vote_cluster_label(const Eigen::MatrixXd &X,
                                    const Eigen::MatrixXd &Y) {
    typedef size_t cluster_index;
    typedef double label_name;
    std::map<cluster_index, std::map<label_name, int> > record_map;
    for (auto i = 0; i < X.rows(); ++i) {
        std::vector<double> dist_vec;
        Eigen::RowVectorXd input_feats = X.row(i);
        for (auto &cluster : _cluster_vec) {
            dist_vec.push_back(calculate_distance(input_feats, cluster.get_mean_vec()));
        }
        std::vector<size_t > sort_idx = GlobalUtils::sort_indexes(dist_vec);
        cluster_index index_tmp = sort_idx[0];
        if (GlobalUtils::has_key(record_map, index_tmp)) {
            label_name label_tmp = Y(i, 0);
            if (GlobalUtils::has_key(record_map[index_tmp], label_tmp)) {
                record_map[index_tmp][label_tmp] += 1;
            } else {
                record_map[index_tmp].insert(std::make_pair(label_tmp, 1));
            }
        } else {
            std::map<label_name, int> tmp = {{Y(i, 0), 1}};
            record_map.insert(std::make_pair(index_tmp, tmp));
        }
    }

    for (auto &record : record_map) {
        cluster_index index = record.first;
        std::map<label_name, int> info_map = record.second;
        label_name label_tmp = WRONG_LABEL;
        int sample_counts = -1;
        for (auto &info : info_map) {
            if (info.second >= sample_counts) {
                sample_counts = info.second;
                label_tmp = info.first;
            }
        }
        _cluster_vec[index].set_cluster_label(label_tmp);
    }
}

double gmmCluster::calculate_distance(const Eigen::RowVectorXd &vectorXd_1,
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

void gmmCluster::test() {
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
    LOG(INFO) << ret << std::endl;
}