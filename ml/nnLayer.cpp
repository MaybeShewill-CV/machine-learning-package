/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: nnLayer.cpp
* Date: 18-4-26 下午2:24
************************************************/

#include "nnLayer.h"

#include <ctime>

#include <cblas.h>
#include <glog/logging.h>

//#define DEBUG

namespace nnLayer_Internal {
    // 计算c = alpha*(a * b) + beta*c矩阵乘法(参照caffe的实现)
    void BLAS_mmul(Eigen::MatrixXd &c, Eigen::MatrixXd &a,
                Eigen::MatrixXd &b, bool aT = false, bool bT = false) {
        CBLAS_TRANSPOSE transA = aT ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE transB = bT ? CblasTrans : CblasNoTrans;

        auto M = static_cast<int>(c.rows());
        auto N = static_cast<int>(c.cols());
        auto K = aT ? static_cast<int>(a.rows()) : static_cast<int>(a.cols());

        float alpha = 1.0f;
        float beta = 1.0f;

        int lda = aT ? K : M;
        int ldb = bT ? N : K;
        int ldc = M;

        cblas_dgemm( CblasColMajor, transA, transB, M, N, K, alpha,
                     a.data(), lda,
                     b.data(), ldb, beta, c.data(), ldc );
    }

    Eigen::MatrixXd random_norm_matrix(const long rows, const long cols) {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::normal_distribution<> random_norm(0, 0.1);
        Eigen::MatrixXd ret(rows, cols);
        for (auto i = 0; i < rows; ++i) {
            for (auto j = 0; j < cols; ++j) {
                ret(i, j) = random_norm(mt);
            }
        }
        return ret;
    }
}

nnLayer::nnLayer(int input_dims, int output_dims, int batch_size) {
    _x = Eigen::MatrixXd::Random(input_dims, batch_size);
    _y = Eigen::MatrixXd::Random(output_dims, batch_size);
    _dx = Eigen::MatrixXd::Random(input_dims, batch_size);
    _dy = Eigen::MatrixXd::Random(output_dims, batch_size);
}

linearTransformLayer::linearTransformLayer(int input_dims, int output_dims, int batch_size) :
        nnLayer(input_dims, output_dims, batch_size){
    _weights = nnLayer_Internal::random_norm_matrix(output_dims, input_dims);
    _bias = Eigen::VectorXd::Zero(output_dims);
}

void linearTransformLayer::forward() {
    _y = _bias.replicate(1, _x.cols());
    nnLayer_Internal::BLAS_mmul(_y, _weights, _x);

#ifdef DEBUG
    LOG(INFO) << "线性变换输入: " << _x << std::endl;
    LOG(INFO) << "线性变换权重: " << _weights << std::endl;
    LOG(INFO) << "线性变换偏置: " << _bias << std::endl;
    LOG(INFO) << "线性变换输出: " << _y << std::endl;
#endif
}

void linearTransformLayer::backward() {
    _dw.setZero();
    nnLayer_Internal::BLAS_mmul(_dw, _dy, _x, false, true);
    _db = _dy.rowwise().sum();
    _dx.setZero();
    nnLayer_Internal::BLAS_mmul(_dx, _weights, _dy, true, false);
}

void linearTransformLayer::resetGrads() {
    _dw = Eigen::MatrixXd::Zero(_weights.rows(), _weights.cols());
    _db = Eigen::VectorXd::Zero(_bias.rows());
}

void linearTransformLayer::applyGrads(const double lr) {
    _bias += lr * _db;
    _weights += lr * _dw;
}

reluLayer::reluLayer(int input_dims, int output_dims, int batch_size) :
        nnLayer(input_dims, output_dims, batch_size){
}

void reluLayer::forward() {
    Eigen::MatrixXd ret(_x.rows(), _x.cols());
    for (auto i = 0; i < _x.rows(); ++i) {
        for (auto j = 0; j < _x.cols(); ++j) {
            ret(i, j) = _relu(_x(i, j));
        }
    }
#ifdef DEBUG
    LOG(INFO) << "relu输入: " << _x << std::endl;
#endif
    _y = ret;
#ifdef DEBUG
    LOG(INFO) << "relu输出: " << _y << std::endl;
#endif
}

void reluLayer::backward() {
    Eigen::MatrixXd relu_grad(_y.rows(), _y.cols());
    for (auto i = 0; i < _y.rows(); ++i) {
        for (auto j =0; j < _y.cols(); ++j) {
            relu_grad(i, j) = _relu_grad(_y(i, j));
        }
    }
    _dx = relu_grad.array() * _dy.array();
}

leakReluLayer::leakReluLayer(int input_dims, int output_dims, int batch_size) :
        nnLayer(input_dims, output_dims, batch_size){}

void leakReluLayer::forward() {
    Eigen::MatrixXd ret(_x.rows(), _x.cols());
    for (auto i = 0; i < _x.rows(); ++i) {
        for (auto j = 0; j < _x.cols(); ++j) {
            ret(i, j) = _leak_relu(_x(i, j));
        }
    }
    _y = ret;
}

void leakReluLayer::backward() {
    Eigen::MatrixXd leak_relu_grad(_y.rows(), _y.cols());
    for (auto i = 0; i < _y.rows(); ++i) {
        for (auto j =0; j < _y.cols(); ++j) {
            leak_relu_grad(i, j) = _leak_relu_grad(_y(i, j));
        }
    }
    _dx = leak_relu_grad.array() * _dy.array();
}

sigmoidLayer::sigmoidLayer(int input_dims, int output_dims, int batch_size) :
        nnLayer(input_dims, output_dims, batch_size){}

void sigmoidLayer::forward() {
    Eigen::MatrixXd ret(_x.rows(), _x.cols());
    for (auto i = 0; i < _x.rows(); ++i) {
        for (auto j = 0; j < _x.cols(); ++j) {
            ret(i, j) = _sigmoid(_x(i, j));
        }
    }
#ifdef DEBUG
    LOG(INFO) << "sigmoid输入: " << _x << std::endl;
#endif
    _y = ret;
#ifdef DEBUG
    LOG(INFO) << "sigmoid输出: " << _y << std::endl;
#endif
}

void sigmoidLayer::backward() {
    Eigen::MatrixXd sigmoid_grad(_y.rows(), _y.cols());
    for (auto i = 0; i < _y.rows(); ++i) {
        for (auto j =0; j < _y.cols(); ++j) {
            sigmoid_grad(i, j) = _sigmoid_grad(_y(i, j));
        }
    }
    _dx = sigmoid_grad.array() * _dy.array();
}

softmaxLayer::softmaxLayer(int input_dims, int output_dims, int batch_size) :
        nnLayer(input_dims, output_dims, batch_size){

}

Eigen::MatrixXd softmaxLayer::_softmax_matrix(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd ret(input.rows(), input.cols());
    Eigen::MatrixXd exp_input = Eigen::MatrixXd(input.array().exp());
    auto exp_input_sum = Eigen::MatrixXd(exp_input).rowwise().sum();
    for (auto row = 0; row < ret.rows(); ++row) {
        for (auto col = 0; col < ret.cols(); ++col) {
            ret(row, col) = exp_input(row, col) / exp_input_sum(row, 0);
        }
    }
    return ret;
}

void softmaxLayer::forward() {
    _y = _softmax_matrix(_x);
}

crossentropyLayer::crossentropyLayer(int input_dims, int output_dims, int batch_size) :
        nnLayer(input_dims, output_dims, batch_size) {
}

void crossentropyLayer::forward() {
#ifdef DEBUG
    LOG(INFO) << "Softmax输入: " << _x.transpose() << std::endl;
#endif
    Eigen::MatrixXd logits = _softmax_matrix(_x.transpose());
#ifdef DEBUG
    LOG(INFO) << "Softmax输出: " << logits << std::endl;
#endif
    _y = logits.transpose();
}

void crossentropyLayer::backward() {
    _dx = _dy - _y;
}

Eigen::MatrixXd crossentropyLayer::_softmax_matrix(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd ret(input.rows(), input.cols());
    Eigen::MatrixXd exp_input = Eigen::MatrixXd(input.array().exp());
    Eigen::MatrixXd exp_input_sum = exp_input.rowwise().sum();
    for (auto row = 0; row < ret.rows(); ++row) {
        for (auto col = 0; col < ret.cols(); ++col) {
            ret(row, col) = exp_input(row, col) / exp_input_sum(row, 0);
        }
    }
    return ret;
}

Eigen::MatrixXd crossentropyLayer::_cross_entropy(const Eigen::MatrixXd &logits,
                                                  const Eigen::MatrixXd &gt_label) {
    Eigen::MatrixXd ret(logits.rows(), logits.cols());
    Eigen::MatrixXd logits_log = Eigen::MatrixXd((logits.array() + 0.000000001).array().log());
    ret = Eigen::MatrixXd(gt_label.array() * logits_log.array());
    ret = -ret;
    return ret;
}
