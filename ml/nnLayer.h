/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: nnLayer.h
* Date: 18-4-26 下午2:24
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_NNLAYER_H
#define MACHINE_LEARNING_PACKAGE_NNLAYER_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

class nnLayer {

public:
    nnLayer() = default;
    virtual ~nnLayer() = default;

    nnLayer(int input_dims, int output_dims, int batch_size);

    // 抽象类接口
    virtual void forward() = 0; // 前向传播
    virtual void backward() = 0; // 后向传播
    virtual void resetGrads() = 0; // 重置梯度
    virtual void applyGrads(double lr) = 0; // 梯度更新

    // 层输入
    Eigen::MatrixXd _x;
    Eigen::MatrixXd _y;

    // 层梯度
    Eigen::MatrixXd _dx;
    Eigen::MatrixXd _dy;
};

class linearTransformLayer : public nnLayer {
public:
    linearTransformLayer() = default;
    ~linearTransformLayer() override = default;

    linearTransformLayer(int input_dims, int output_dims, int batch_size);

    void forward() override ;
    void backward() override ;
    void resetGrads() override ;
    void applyGrads(double lr) override ;

    // 获取成员变量
    Eigen::MatrixXd get_weights() {return _weights;};
    Eigen::MatrixXd get_bias() {return _bias;};

    Eigen::MatrixXd get_weights_grad() {return _dw;};
    Eigen::MatrixXd get_bias_grad() {return _db;};

private:
    // 线性变换参数
    Eigen::MatrixXd _weights;
    Eigen::VectorXd _bias;

    // 线性变换参数梯度
    Eigen::MatrixXd _dw;
    Eigen::MatrixXd _db;
};

class reluLayer : public nnLayer {
public:
    reluLayer() = default;
    ~reluLayer() override = default;

    reluLayer(int input_dims, int output_dims, int batch_size);

    void forward() override ;
    void backward() override ;
    void resetGrads() override {};
    void applyGrads(double lr) override {};

private:
    double _relu(double input) {return (input > 0) ? input : 0;};
    double _relu_grad(double input) {return (input > 0) ? 1 : 0;};
};

// 测试使用 所以没有backward接口
class softmaxLayer : public nnLayer {
public:
    softmaxLayer() = default;
    ~softmaxLayer() override = default;

    softmaxLayer(int input_dims, int output_dims, int batch_size);

    void forward() override ;
    void backward() override {};
    void resetGrads() override {};
    void applyGrads(double lr) override {};

private:
    Eigen::MatrixXd _softmax_matrix(const Eigen::MatrixXd &input);
};

// 训练使用 含有backward接口
class crossentropyLayer : public nnLayer {
public:
    crossentropyLayer() = default;
    ~crossentropyLayer() override = default;

    crossentropyLayer(int input_dims, int output_dims, int batch_size);

    void forward() override ;
    void backward() override ;
    void resetGrads() override {};
    void applyGrads(double lr) override {};

    Eigen::MatrixXd _cross_entropy(const Eigen::MatrixXd &logits,
                                  const Eigen::MatrixXd &gt_label);

private:
    Eigen::MatrixXd _softmax_matrix(const Eigen::MatrixXd &input);
};

#endif //MACHINE_LEARNING_PACKAGE_NNLAYER_H
