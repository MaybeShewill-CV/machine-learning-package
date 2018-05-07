/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: svmClassifier.h
* Date: 18-5-3 下午5:48
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_SVMCLASSIFIER_H
#define MACHINE_LEARNING_PACKAGE_SVMCLASSIFIER_H

#include <MLBase.h>

#include <stack>

enum KERNEL_TYPE {
    LINEAR,
    POLYNOMIAL,
    GAUSSIAN,
    RBF
};

class svmClassifier : public MLBase {
public:
    svmClassifier() = default;
    ~svmClassifier() override = default;

    svmClassifier(int iter_times, double C, double tol, double bias, KERNEL_TYPE kernel_type);

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;

    Eigen::VectorXd get_lagrangian_mul_coffecient() {return _lagrangian_mul_coffecient;};

private:
    double _bias = 0.0;
    Eigen::VectorXd _lagrangian_mul_coffecient;
    double _C = 1.0;
    double _tol = 0.00000001;
    int _max_iter_times = 1000;
    KERNEL_TYPE _kernelType = RBF;
    Eigen::MatrixXd _x;
    Eigen::MatrixXd _y;
    std::stack<bool> _process_status_stack;

    // 初始化拉格朗日乘常数
    void init_lagrangian_mul_coffecient(const Eigen::MatrixXd &Y);
    // 计算上下界
    double compute_L(const Eigen::MatrixXd &Y, int index_1, int index_2);
    double compute_H(const Eigen::MatrixXd &Y, int index_1, int index_2);
    // 前向传播
    double forward_train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
                         const Eigen::VectorXd &input_vec);
    // 内积
    double inner_product(const Eigen::VectorXd &input_1, const Eigen::VectorXd &input_2);
    // smo参数优化
    void simplified_smo(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    // smo参数优化终止条件
    bool terminate_smo();
    // 核函数
    double linear_kernel_func(const Eigen::VectorXd &input_1, const Eigen::VectorXd &input_2);
    double polynomial_kernel_func(const Eigen::VectorXd &input_1, const Eigen::VectorXd &input_2, int d);
    double gaussian_kernel_func(const Eigen::VectorXd &input_1, const Eigen::VectorXd &input_2);
    double rbf_kernel_func(const Eigen::VectorXd &input_1, const Eigen::VectorXd &input_2);
};

#endif //MACHINE_LEARNING_PACKAGE_SVMCLASSIFIER_H
