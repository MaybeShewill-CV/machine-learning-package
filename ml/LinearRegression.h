/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: linearRegression.h
* Date: 18-4-3 下午3:00
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_LINEARREGRESSION_H
#define MACHINE_LEARNING_PACKAGE_LINEARREGRESSION_H

#include <MLBase.h>

class LinearRegression:virtual public MLBase{
public:
    LinearRegression();
    ~LinearRegression() override ;

    LinearRegression(double lr, int iter_nums);

    void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) override;
    void predict(const Eigen::MatrixXd& X, Eigen::MatrixXd& Y) override ;

private:
    void init_weights(int feats_dims, int label_dims, Eigen::MatrixXd& kernel);
    double init_bias();
    Eigen::MatrixXd linear_transform(
            const Eigen::MatrixXd& X, const Eigen::MatrixXd& kernal,
                                      double bias);
    double compute_square_loss(const Eigen::MatrixXd& label, const Eigen::MatrixXd& predicts);
    Eigen::MatrixXd compuet_kernel_gradient(
            const Eigen::MatrixXd& preds, const Eigen::MatrixXd& label, const Eigen::MatrixXd& X);
    double compuet_bias_gradient(
            const Eigen::MatrixXd& preds, const Eigen::MatrixXd& label);

    double lr = 0.0001;
    int iter_nums = 10000000;
    Eigen::MatrixXd kernel;
    double bias;

};


#endif //MACHINE_LEARNING_PACKAGE_LINEARREGRESSION_H
