/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: logisticRegression.h
* Date: 18-4-10 下午2:39
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_LOGISTICREGRESSION_H
#define MACHINE_LEARNING_PACKAGE_LOGISTICREGRESSION_H

#include <MLBase.h>

class logisticRegression: public MLBase {
public:
    logisticRegression() = default;
    ~logisticRegression() override = default;

    enum GDTYPE {
        SGD,
        GD
    };

    logisticRegression(double learning_rate, int iter_times);

    void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) override ;
    void predict(const Eigen::MatrixXd& X, Eigen::MatrixXd& RET) override ;

    Eigen::RowVectorXd get_kernel() {return kernel;};

private:
    double lr = 0.1;
    int iter_times = 10000;
    GDTYPE gdtype = SGD;
    Eigen::RowVectorXd kernel;

    double sigmoid(double z) { return 1.0 / (1 + exp(-z)); }
    Eigen::RowVectorXd sigmoid(const Eigen::RowVectorXd &X);
    void init_kernel(int feats_dims);
    void normalize_feats(Eigen::MatrixXd &X);

    Eigen::RowVectorXd forward(Eigen::MatrixXd X);
    double compute_loss(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    Eigen::RowVectorXd compute_kernel_gradient(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    void forward_backward_update(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
};


#endif //MACHINE_LEARNING_PACKAGE_LOGISTICREGRESSION_H
