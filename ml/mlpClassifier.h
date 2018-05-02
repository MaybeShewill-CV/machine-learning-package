/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: mlpClassifier.h
* Date: 18-4-26 下午7:02
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_MLPCLASSIFIER_H
#define MACHINE_LEARNING_PACKAGE_MLPCLASSIFIER_H

#include <deque>
#include <stack>

#include <MLBase.h>
#include <nnLayer.h>

class mlpClassifier : public MLBase {
public:
    mlpClassifier() = default;
    ~mlpClassifier() override;

    mlpClassifier& operator = (const mlpClassifier &other);

    mlpClassifier(int class_nums, int max_iter_times, double lr, int batch_size, int input_dims);

    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) override ;
    void predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) override ;

    std::deque<nnLayer*> get_mlp_layer() {return mlp_layer;};

private:
    int _class_nums = 0;
    int _max_iter_times = 0;
    double _lr = 0;
    int _batch_size = 0;
    // 保存网络结构
    std::deque<nnLayer*> mlp_layer;

    // one_hot编码标签向量(原始标签向量需要从0开始标记并且是连续数)
    Eigen::MatrixXd one_hot_encode(const Eigen::MatrixXd &input);
    // 一个epoch的训练,主要设计batch数据的获取
    void fit_one_epoch(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, int epoch);
    // 获取batch数据
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> get_batch_data(const Eigen::MatrixXd &input_data,
                                                               const Eigen::MatrixXd &input_label,
                                                               std::stack<long> &index_stack);
    // mlp前向传播
    Eigen::MatrixXd mlp_forward(const Eigen::MatrixXd &X);
    // bp更新网络权重
    void mlp_backward_update(const Eigen::MatrixXd &Y);

};

#endif //MACHINE_LEARNING_PACKAGE_MLPCLASSIFIER_H
