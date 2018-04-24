/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: nnclassifier.h
* Date: 18-4-24 下午3:01
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_NNCLASSIFIER_H
#define MACHINE_LEARNING_PACKAGE_NNCLASSIFIER_H

#include <MLBase.h>

#include <vector>

enum ACTIVATION_TYPE {
    SIGMOID,
    RELU
};

typedef long rows;
typedef long cols;

class nnclassifier : public MLBase {
public:
    nnclassifier() = default;
    ~nnclassifier() override = default;

    nnclassifier(int class_nums, int layer_nums, int hidden_uint_nums,
                 ACTIVATION_TYPE activation_type);

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;

    void test();

private:
    int _layer_nums = 0;
    int _hidden_unit_nums = 0;
    int _class_nums = 0;
    ACTIVATION_TYPE _activationType = RELU;
    std::vector<Eigen::MatrixXd> _kernel_vec;
    std::vector<std::pair<rows, cols> > _kernel_size_vec;

    // 初始化参数
    void init_kernel(const Eigen::MatrixXd &X);
    // 前向传播
    Eigen::MatrixXd forward_broadcast(const Eigen::MatrixXd &X);
    // 扩展特征矩阵最后一列为1(为了将bias集成进矩阵乘法中)
    Eigen::MatrixXd expand_feats_matrix(const Eigen::MatrixXd &X);
    // 激活函数
    double activate_unit(double input);

};


#endif //MACHINE_LEARNING_PACKAGE_NNCLASSIFIER_H
