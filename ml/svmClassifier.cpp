/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: svmClassifier.cpp
* Date: 18-5-3 下午5:48
************************************************/

#include "svmClassifier.h"

#include <ctime>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#define DEBUG

namespace svm_internal {
    std::default_random_engine E(time(nullptr));

    template <class T>
    void clear_stack(std::stack<T> _stack) {
        while (!_stack.empty()) {
            _stack.pop();
        }
    }
}

svmClassifier::svmClassifier(const int iter_times, const double C, const double tol,
                             const double bias, const KERNEL_TYPE kernel_type) :
        _bias(bias), _C(C), _tol(tol), _max_iter_times(iter_times),
        _kernelType(kernel_type) {}

void svmClassifier::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // 初始化特征空间矩阵和标签向量
    _x = X;
    _y = Y;
    // 初始化模型拉格朗日乘常数
    init_lagrangian_mul_coffecient(Y);
    // smo算法更新参数
    simplified_smo(X, Y);
}

void svmClassifier::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::MatrixXd ret(X.rows(), 1);
    for (auto i = 0; i < X.rows(); ++i) {
        double forward_ret = forward_train(_x, _y, X.row(i));
        if (forward_ret >= 0) {
            ret(i, 0) = 1;
        } else {
            ret(i, 0) = -1;
        }
    }
    RET = ret;
}

// 根据CS 229, Autumn 2009　The Simplified SMO Algorithm实现的Simplified SMO Algorithm
void svmClassifier::simplified_smo(const Eigen::MatrixXd &X,
                                   const Eigen::MatrixXd &Y) {
    auto loop_times = 0;
    auto loop_times_2 = 1;
    if (!_process_status_stack.empty()) {
        svm_internal::clear_stack(_process_status_stack);
    }

    while (loop_times < _max_iter_times && !terminate_smo()) {
        auto changed_alpha_nums = 0;
        for (auto i = 0; i < X.rows(); ++i) {
            auto error_i = forward_train(X, Y, X.row(i)) - Y(i, 0);
            if ((Y(i, 0) * error_i < -_tol && _lagrangian_mul_coffecient(i) < _C) ||
                    (Y(i, 0) * error_i > _tol && _lagrangian_mul_coffecient(i) > 0)) {
                std::vector<long> index_vec;
                for (auto j = 0; j < X.rows(); ++j) {
                    if (j == i) {
                        continue;
                    }
                    index_vec.push_back(j);
                }
                std::uniform_int_distribution<size_t > random_index(0, index_vec.size() - 1);
                auto select_index = random_index(svm_internal::E);
                auto error_j = forward_train(X, Y, X.row(index_vec[select_index])) -
                        Y(index_vec[select_index], 0);
                auto a_i = _lagrangian_mul_coffecient(i);
                auto a_j = _lagrangian_mul_coffecient(index_vec[select_index]);
                auto y_i = Y(i, 0);
                auto y_j = Y(index_vec[select_index], 0);
                auto L = compute_L(Y, i, static_cast<int>(index_vec[select_index]));
                auto H = compute_H(Y, i, static_cast<int>(index_vec[select_index]));
                if (std::abs(L - H) < _tol) {
#ifdef DEBUG
                    LOG(INFO) << "[Epoch/Iter]: " << "[" << loop_times_2 << "/" << i << "]:" << std::endl;
                    LOG(INFO) << "---参数未更新" << std::endl;
#endif
                    _process_status_stack.push(false);
                    continue;
                }
                Eigen::VectorXd sample_i = X.row(i);
                Eigen::VectorXd sample_j = X.row(index_vec[select_index]);
                double eta = 2 * inner_product(sample_i, sample_j) -
                        inner_product(sample_i, sample_i) - inner_product(sample_j, sample_j);
                if (eta >= 0) {
#ifdef DEBUG
                    LOG(INFO) << "[Epoch/Iter]: " << "[" << loop_times_2 << "/" << i << "]:" << std::endl;
                    LOG(INFO) << "---参数未更新" << std::endl;
#endif
                    _process_status_stack.push(false);
                    continue;
                }
                auto a_j_clip = a_j;
                a_j_clip -= y_j * (error_i - error_j) / eta;
                if (a_j_clip >= H) {
                    a_j_clip = H;
                } else if (a_j_clip < L) {
                    a_j_clip = L;
                }

                if (std::abs(a_j - a_j_clip) < _tol) {
#ifdef DEBUG
                    LOG(INFO) << "[Epoch/Iter]: " << "[" << loop_times_2 << "/" << i << "]:" << std::endl;
                    LOG(INFO) << "---参数未更新" << std::endl;
#endif
                    _process_status_stack.push(false);
                    continue;
                }
                auto s = y_i * y_j;
                auto a_i_new = a_i + s * (a_j - a_j_clip);
                auto b_i = _bias - error_i - y_i * (a_i_new - a_i) * inner_product(sample_i, sample_i)
                           - y_j * (a_j_clip - a_j) * inner_product(sample_j, sample_j);
                auto b_j = _bias - error_j - y_i * (a_i_new - a_i) * inner_product(sample_i, sample_i)
                           - y_j * (a_j_clip - a_j) * inner_product(sample_j, sample_j);
                auto b = 0.0;
                if (a_i_new > 0 && a_i_new < _C) {
                    b = b_i;
                } else if (a_j_clip > 0 && a_j_clip < _C) {
                    b = b_j;
                } else {
                    b = (b_i + b_j) / 2;
                }
                changed_alpha_nums += 1;
                // 更新参数
                auto b_old = _bias;
                _bias = b;
                _lagrangian_mul_coffecient(i) = a_i_new;
                _lagrangian_mul_coffecient(index_vec[select_index]) = a_j_clip;

#ifdef DEBUG
                // 输出参数更新信息
                LOG(INFO) << "[Epoch/Iter]: " << "[" << loop_times_2 << "/" << i << "]:" << std::endl;
                LOG(INFO) << "---a_i_old: " << a_i << ", a_i_new: " << a_i_new << std::endl;
                LOG(INFO) << "---a_j_old: " << a_j << ", a_j_new: " << a_j_clip << std::endl;
                LOG(INFO) << "---b_old: " << b_old << ", b_new: " << b << std::endl;
#endif
                _process_status_stack.push(true);
            } else {
#ifdef DEBUG
                LOG(INFO) << "[Epoch/Iter]: " << "[" << loop_times_2 << "/" << i << "]:" << std::endl;
                LOG(INFO) << "---参数未更新" << std::endl;
#endif
                _process_status_stack.push(false);
            }
        }
        loop_times_2 += 1;
        if (changed_alpha_nums == 0) {
            loop_times += 1;
        } else {
            loop_times = 0;
        }
    }
}

bool svmClassifier::terminate_smo() {
    if (_process_status_stack.empty()) {
        return false;
    }

    // 连续20次没有进行参数更新则认为参数已经更新完毕,停止smo更新参数(这里可以调整smo更新终止策略)
    bool terminate = true;
    auto loop_times = _process_status_stack.size() >= 20 ? 20 : _process_status_stack.size();
    while (loop_times--) {
        if (_process_status_stack.top()) {
            _process_status_stack.pop();
        } else {
            return false;
        }
    }
    return terminate;
}

double svmClassifier::forward_train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
                                    const Eigen::VectorXd &input_vec) {
    auto ret = 0.0;
    for (auto i = 0; i < X.rows(); ++i) {
        ret += _lagrangian_mul_coffecient(i) * Y(i, 0) * inner_product(X.row(i), input_vec);
    }
    ret += _bias;
    return ret;
}

double svmClassifier::inner_product(const Eigen::VectorXd &input_1,
                                    const Eigen::VectorXd &input_2) {
    auto ret = 0.0;
    switch (_kernelType) {
        case LINEAR: {
            ret = linear_kernel_func(input_1, input_2);
            break;
        }
        case POLYNOMIAL: {
            ret = polynomial_kernel_func(input_1, input_2, 2);
            break;
        }
        case GAUSSIAN: {
            ret = gaussian_kernel_func(input_1, input_2);
            break;
        }
        case RBF: {
            ret = rbf_kernel_func(input_1, input_2);
            break;
        }
    }
    return ret;
}

void svmClassifier::init_lagrangian_mul_coffecient(const Eigen::MatrixXd &Y) {
    auto coffecient_nums = Y.rows();

    double pos_sum = 0.0;
    double neg_sum = 0.0;

    std::vector<int> pos_index;
    std::vector<int> neg_index;

    for (auto i = 0; i < coffecient_nums; ++i) {
        auto y_i = Y(i, 0);
        if (y_i > 0) {
            pos_sum += y_i;
            pos_index.push_back(i);
        } else {
            neg_index.push_back(i);
        }
    }

    neg_sum = pos_sum;
    std::vector<double> pos_coffecient;
    std::vector<double> neg_coffecient;

    for (size_t i = 0; i < pos_index.size() - 1; ++i) {
        std::uniform_real_distribution<double> random_d(0, pos_sum);
        auto random_coffecient = random_d(svm_internal::E);
        pos_coffecient.push_back(random_coffecient);
        pos_sum -= random_coffecient;
    }
    pos_coffecient.push_back(pos_sum);

    for (size_t i = 0; i < neg_index.size() - 1; ++i) {
        std::uniform_real_distribution<double> random_d(0, neg_sum);
        auto random_coffecient = random_d(svm_internal::E);
        neg_coffecient.push_back(random_coffecient);
        neg_sum -= random_coffecient;
    }
    neg_coffecient.push_back(neg_sum);

    std::shuffle(pos_coffecient.begin(), pos_coffecient.end(), std::default_random_engine(time(nullptr)));
    std::shuffle(neg_coffecient.begin(), neg_coffecient.end(), std::default_random_engine(time(nullptr)));

    Eigen::VectorXd mul_coffecient(Y.rows());
    for (size_t i = 0; i < pos_index.size(); ++i) {
        mul_coffecient(pos_index[i]) = pos_coffecient[i];
    }
    for (size_t i = 0; i < neg_index.size(); ++i) {
        mul_coffecient(neg_index[i]) = neg_coffecient[i];
    }

#ifdef DEBUG
    auto dot_sum = mul_coffecient.dot(Y.col(0));
    LOG(INFO) << "拉格朗日乘常数约束检查值为: " << dot_sum << std::endl;
#endif
    _lagrangian_mul_coffecient = mul_coffecient;
//    _lagrangian_mul_coffecient = Eigen::VectorXd::Zero(Y.rows());
}

double svmClassifier::compute_H(const Eigen::MatrixXd &Y, const int index_1, const int index_2) {
    auto y_1 = Y(index_1, 0);
    auto y_2 = Y(index_2, 0);
    auto a_1 = _lagrangian_mul_coffecient(index_1);
    auto a_2 = _lagrangian_mul_coffecient(index_2);

    if (y_1 == y_2) {
        return std::min(_C, a_1 + a_2);
    } else {
        return std::min(_C, _C + a_2 - a_1);
    }
}

double svmClassifier::compute_L(const Eigen::MatrixXd &Y, const int index_1, const int index_2) {
    auto y_1 = Y(index_1, 0);
    auto y_2 = Y(index_2, 0);
    auto a_1 = _lagrangian_mul_coffecient(index_1);
    auto a_2 = _lagrangian_mul_coffecient(index_2);

    if (y_1 == y_2) {
        return std::max(0.0, a_1 + a_2 - _C);
    } else {
        return std::max(0.0, a_2 - a_1);
    }
}

double svmClassifier::linear_kernel_func(const Eigen::VectorXd &input_1,
                                         const Eigen::VectorXd &input_2) {
    return input_1.dot(input_2);
}

double svmClassifier::polynomial_kernel_func(const Eigen::VectorXd &input_1,
                                             const Eigen::VectorXd &input_2, int d) {
    return std::pow(input_1.dot(input_2), d);
}

double svmClassifier::gaussian_kernel_func(const Eigen::VectorXd &input_1,
                                           const Eigen::VectorXd &input_2) {
    Eigen::VectorXd diff = input_1 - input_2;
    diff = diff.array().pow(2);
    auto diff_sum = diff.sum();
    diff_sum = std::sqrt(diff_sum);
    double sigma = 1;
    return std::exp(-diff_sum / (2 * std::pow(sigma, 1)));
}

double svmClassifier::rbf_kernel_func(const Eigen::VectorXd &input_1,
                                      const Eigen::VectorXd &input_2) {
    Eigen::VectorXd diff = input_1 - input_2;
    diff = diff.array().pow(2);
    auto diff_sum = diff.sum();
    diff_sum = std::sqrt(diff_sum);
    double gamma = 0.5;
    return std::exp(-gamma * diff_sum);
}
