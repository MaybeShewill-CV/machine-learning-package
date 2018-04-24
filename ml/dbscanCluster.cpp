/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: dbscanCluster.cpp
* Date: 18-4-23 下午4:26
************************************************/

#include "dbscanCluster.h"

#include <glog/logging.h>

#include <globalUtils.h>

#define DEBUG

namespace dbscanCluster_internal {
    std::default_random_engine E(static_cast<u_long >(time(nullptr)));

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
    }

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
                           std::vector<dbscan_cluster> &cluster_vec,
                           std::map<double, cv::Scalar> color_map) {
        // 画出原型向量的位置
        for (size_t i = 0; i < cluster_vec.size(); ++i) {
            Eigen::RowVectorXd prototype_vec = cluster_vec[i].get_seed_feats();
            cv::Point pt(static_cast<int>(prototype_vec[0] * 500),
                         500 - static_cast<int>(prototype_vec[1] * 500));
            auto line_length = 40;
            draw_cross(image, pt, line_length, color_map[i]);
        }

        // 画出数据点分布
        for (size_t i = 0; i < cluster_vec.size(); ++i) {
            Eigen::MatrixXd sample_matrix = cluster_vec[i].get_sample_matrix();
            for (auto j = 0; j < sample_matrix.rows(); ++j) {
                cv::Point pt(static_cast<int>(sample_matrix(j, 0) * 500),
                             500 - static_cast<int>(sample_matrix(j, 1) * 500));
                cv::circle(image, pt, 4, color_map[i], 5, CV_FILLED);
            }
        }
    }
}

dbscanCluster::dbscanCluster(const int core_obj_nums_threshold, const double dist_threshold,
                             DISTANCE_TYPE distance_type) :
        _core_obj_nums_threshold(core_obj_nums_threshold),
        _dist_threshold(dist_threshold),
        _distanceType(distance_type){}

void dbscanCluster::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 初始化核心对象集合
    if (!is_core_obj_validly_init(X)) {
        return;
    }
    // 初始化未访问对象集合
    init_no_visit_sample_set(X);

    int loop_time = 0;
    while (true) {
        LOG(INFO) << "开始生长聚类簇: " << loop_time + 1 << std::endl;
        // 随机选择种子核心对象
        std::uniform_int_distribution<sample_index > u(0, _core_obj_map.size() - 1);
        auto index_tmp = u(dbscanCluster_internal::E);
        auto iter = _core_obj_map.begin();
        for (auto i = 0; i < index_tmp; ++i) {
            iter++;
        }
        auto seed_sample_index  = iter->first;
        std::vector<sample_index > has_been_selected;
        auto reachable_cluster_ret = find_density_reachable_set(X, seed_sample_index, has_been_selected);
        Eigen::RowVectorXd tmp(1, reachable_cluster_ret.size());
        int index = 0;
        for (auto &ele : reachable_cluster_ret) {
            tmp(0, index) = ele;
            index++;
        }
        LOG(INFO) << "--聚类簇" << loop_time + 1 << "的密度可达对象集合为: " << tmp << std::endl;
        update_cluster(reachable_cluster_ret);
        Eigen::MatrixXd sample_matrix(reachable_cluster_ret.size(), X.cols());
        auto matrix_row_index = 0;
        for (auto &sample_index : reachable_cluster_ret) {
            sample_matrix.row(matrix_row_index) = X.row(sample_index);
            matrix_row_index++;
        }
        dbscan_cluster cluster_tmp(X.row(seed_sample_index), sample_matrix);
        cluster_tmp.set_cluster_label(vote_cluster_label(reachable_cluster_ret, Y));
        _cluster_vec.push_back(cluster_tmp);
        loop_time++;
        if (_core_obj_map.empty()) {
            LOG(INFO) << "聚类簇聚合完毕" << std::endl;
            break;
        }
    }
#ifdef DEBUG
    vis_cluster(X, Y);
#endif
}

void dbscanCluster::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::MatrixXd ret(X.rows(), 1);
    for (auto i = 0; i < X.rows(); ++i) {
        ret(i, 0) = predict_one_sample(X.row(i));
    }
    RET = ret;
}

double dbscanCluster::predict_one_sample(const Eigen::RowVectorXd &input_vec) {
    std::vector<double> dist_vec;
    for (auto &cluster : _cluster_vec) {
        dist_vec.push_back(calculate_distance(input_vec, cluster.get_seed_feats()));
    }
    std::vector<size_t > sort_idx = GlobalUtils::sort_indexes(dist_vec);
    return _cluster_vec[sort_idx[0]].get_cluster_label();
}

void dbscanCluster::init_core_obj_set(const Eigen::MatrixXd &X) {
    std::map<sample_index, std::set<sample_index > > core_obj_index;
    for (auto i = 0; i < X.rows(); ++i) {
        std::set<sample_index > neighbor_set;
        neighbor_set.insert(i);
        for (auto j = 0; j < X.rows(); ++j) {
            if (i == j) {
                continue;
            } else {
                if (calculate_distance(X.row(i), X.row(j)) <= _dist_threshold) {
                    neighbor_set.insert(j);
                }
            }
        }
        if (static_cast<int>(neighbor_set.size()) >= _core_obj_nums_threshold) {
            core_obj_index.insert(std::make_pair(i, neighbor_set));
        }
    }
    _core_obj_map = core_obj_index;
}

bool dbscanCluster::is_core_obj_validly_init(const Eigen::MatrixXd &X) {
    init_core_obj_set(X);

    // 核心对象集合为空,初始化失败
    if (_core_obj_map.empty()) {
        LOG(ERROR) << "初始化核心对象集合失败, 是否设置过小的距离阈值或者过大的邻域元素个数阈值" << std::endl;
        return false;
    }

    // 原始数据中每个元素都为核心对象且其邻域都是整个原始数据空间,初始化失败
    if (_core_obj_map.size() == static_cast<size_t >(X.rows())) {
        LOG(ERROR) << "初始化核心对象集合失败, 是否设置过大的距离阈值" << std::endl;
        return false;
    }

    LOG(INFO) << "初始化核心对象集合成功" << std::endl;
    return true;
}

void dbscanCluster::init_no_visit_sample_set(const Eigen::MatrixXd &X) {
    for (auto i = 0; i < X.rows(); ++i) {
        _no_visit_sample_set.insert(i);
    }
}

std::set<sample_index > dbscanCluster::find_density_reachable_set(
        const Eigen::MatrixXd &X, const sample_index seed_index,
        std::vector<sample_index > sample_has_been_selected) {
    std::set<sample_index > reachable_set;
    // 在还没有被访问的样本集合中寻找密度可达样本
    for (auto &neighbor_index : _core_obj_map[seed_index]) {
        // 如果样本已经被访问过则跳过该样本
        if (GlobalUtils::has_elements(sample_has_been_selected, neighbor_index)) {
            continue;
        }
        // 如果该样本是核心对象则递归查找核心对象的密度可达样本
        else if (GlobalUtils::has_key(_core_obj_map, neighbor_index)) {
            sample_has_been_selected.push_back(neighbor_index);
            reachable_set.insert(neighbor_index);
            std::set<sample_index > union_ret;
            std::set<sample_index > find_recur = find_density_reachable_set(X, neighbor_index,
                                                                            sample_has_been_selected);
            std::set_union(find_recur.begin(),
                           find_recur.end(),
                           reachable_set.begin(), reachable_set.end(),
                           std::inserter(union_ret, union_ret.begin()));
            reachable_set = union_ret;
        } // 如果该样本不是核心对象则添加进密度可达样本
        else {
            reachable_set.insert(neighbor_index);
        }
    }
    return reachable_set;
}

void dbscanCluster::update_cluster(const std::set<sample_index> &reachable_set) {
    std::map<sample_index, std::set<sample_index > > new_core_obj_map;
    std::set<sample_index > new_no_visit_sample_set;
    for (auto &core_obj : _core_obj_map) {
        if (GlobalUtils::has_elements(reachable_set, core_obj.first)) {
            continue;
        } else {
            new_core_obj_map.insert(core_obj);
        }
    }

    for (auto &sample : _no_visit_sample_set) {
        if (GlobalUtils::has_elements(reachable_set, sample)) {
            continue;
        } else {
            new_no_visit_sample_set.insert(sample);
        }
    }
    _core_obj_map = new_core_obj_map;
    _no_visit_sample_set = new_no_visit_sample_set;
}

double dbscanCluster::vote_cluster_label(const std::set<sample_index> &sample_set,
                                         const Eigen::MatrixXd &Y) {
    std::map<label_name, int> count_map;
    for (auto &sample_index : sample_set) {
        if (GlobalUtils::has_key(count_map, Y(sample_index, 0))) {
            count_map[Y(sample_index, 0)] += 1;
        } else {
            count_map[Y(sample_index, 0)] = 1;
        }
    }
    int sample_nums = -1;
    label_name label = WRONG_LABEL;
    for (auto &info : count_map) {
        if (info.second >= sample_nums) {
            sample_nums = info.second;
            label = info.first;
        }
    }
    return label;
}

void dbscanCluster::vis_cluster(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 画出原始数据分布
    cv::Mat origin_data_distribution(500, 500, CV_8UC3);
    dbscanCluster_internal::vis_origin_feats_distribute(origin_data_distribution, X, Y);

    // 画出聚类簇分布
    auto color_map = dbscanCluster_internal::init_color_map(static_cast<int>(_cluster_vec.size() + 1));
    cv::Mat cluster_data_distribution(500, 500, CV_8UC3);
    dbscanCluster_internal::vis_prototype_vec(cluster_data_distribution, X, _cluster_vec, color_map);

    // 画出还没有被访问的元素
    for (auto &sample_index : _no_visit_sample_set) {
        cv::Point center(static_cast<int>(X(sample_index, 0) * 500),
                         500 - static_cast<int>(X(sample_index, 1) * 500));
        cv::circle(cluster_data_distribution, center, 4, (--color_map.end())->second, 5, CV_FILLED);
    }

    cv::imshow("原始数据分布", origin_data_distribution);
    cv::imshow("DBSCAN聚类簇结果", cluster_data_distribution);
    cv::waitKey();
}

double dbscanCluster::calculate_distance(const Eigen::RowVectorXd &vectorXd_1,
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

void dbscanCluster::test() {
    Eigen::MatrixXd X(30, 2);
    Eigen::MatrixXd Y(30, 1);
    X << 0.697, 0.460, 0.774, 0.376, 0.634, 0.264, 0.608, 0.318, 0.556, 0.215, 0.403, 0.237, 0.481, 0.149, 0.437,
            0.211, 0.666, 0.091, 0.243, 0.267, 0.245, 0.057, 0.343, 0.099, 0.639, 0.161, 0.657, 0.198, 0.360, 0.370,
            0.593, 0.042, 0.719, 0.103, 0.359, 0.188, 0.339, 0.241, 0.282, 0.257, 0.748, 0.232, 0.714, 0.346, 0.483,
            0.312, 0.478, 0.437, 0.525, 0.369, 0.751, 0.489, 0.532, 0.472, 0.473, 0.376, 0.725, 0.445, 0.446, 0.459;
    Y << 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    fit(X, Y);
}