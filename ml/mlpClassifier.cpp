/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: mlpClassifier.cpp
* Date: 18-4-26 下午7:02
************************************************/

#include "mlpClassifier.h"

#include <glog/logging.h>

#include <mnistDataloder.h>

//#define DEBUG

mlpClassifier::mlpClassifier(const int class_nums, const int max_iter_times,
                             const double lr, const int batch_size, const int input_dims) :
        _class_nums(class_nums), _max_iter_times(max_iter_times),
        _lr(lr), _batch_size(batch_size){
    // 初始化mlp网络结构
    mlp_layer.push_back(new linearTransformLayer(input_dims, 256, _batch_size));
    mlp_layer.push_back(new reluLayer(256, 256, _batch_size));
    mlp_layer.push_back(new linearTransformLayer(256, 256, _batch_size));
    mlp_layer.push_back(new reluLayer(256, 256, _batch_size));
    mlp_layer.push_back(new linearTransformLayer(256, 100, _batch_size));
    mlp_layer.push_back(new reluLayer(100, 100, _batch_size));
    mlp_layer.push_back(new linearTransformLayer(100, _class_nums, _batch_size));
    mlp_layer.push_back(new crossentropyLayer(_class_nums, _class_nums, _batch_size));
}

mlpClassifier& mlpClassifier::operator=(const mlpClassifier &other) {
    // 检查自赋值
    if (this != &other) {
        this->_class_nums = other._class_nums;
        this->_batch_size = other._batch_size;
        this->_max_iter_times = other._max_iter_times;
        this->_lr = other._lr;
        this->mlp_layer = other.mlp_layer;
    }
    return *this;
}

mlpClassifier::~mlpClassifier() {
//    for (auto &layer : mlp_layer) {
//        delete(layer);
//    }
    mlp_layer.clear();
    std::deque<nnLayer*>().swap(mlp_layer);
}

void mlpClassifier::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    // one_hot编码标签
    Eigen::MatrixXd one_hot_label = one_hot_encode(Y);
    // BP训练更新损失
    for (auto loop_time = 0; loop_time < _max_iter_times; ++loop_time) {
        // 训练一个epoch
        fit_one_epoch(X, one_hot_label, loop_time + 1);
    }
}

void mlpClassifier::predict(const Eigen::MatrixXd &X, Eigen::MatrixXd &RET) {
    Eigen::MatrixXd softmax_out = mlp_forward(X);
    Eigen::MatrixXd ret(X.rows(), 1);
    for (auto row = 0; row < softmax_out.rows(); ++row) {
        Eigen::Index row_idx, col_idx;
        softmax_out.row(row).maxCoeff(&row_idx, &col_idx);
        ret(row, 0) = static_cast<double>(col_idx);
    }
    RET = ret;
}

void mlpClassifier::fit_one_epoch(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
                                  const int epoch) {
    // 随机打乱样本顺序
    std::stack<long> index_stack;
    std::vector<long> index_vec;
    for (auto i = 0; i < X.rows(); ++i) {
        index_vec.push_back(i);
    }
    std::shuffle(index_vec.begin(), index_vec.end(),
                 std::default_random_engine(static_cast<ulong>(time(nullptr))));
    for (auto &index : index_vec) {
        index_stack.push(index);
    }
    auto iter_times = std::ceil(static_cast<double>(X.rows() / _batch_size));

    for (auto i = 0; i < iter_times; ++i) {
        auto batch_data = get_batch_data(X, Y, index_stack);
        Eigen::MatrixXd input_data = batch_data.first;
        Eigen::MatrixXd input_label = batch_data.second;
        // 前向传播计算交叉熵损失
        mlp_layer[0]->_x = input_data.transpose();
        size_t layer_index = 0;
        while (layer_index < mlp_layer.size()) {
            mlp_layer[layer_index]->forward();
            if (layer_index + 1 < mlp_layer.size()) {
                mlp_layer[layer_index + 1]->_x = mlp_layer[layer_index]->_y;
            }
            layer_index++;
        }
        Eigen::MatrixXd cross_entropy_matrix =
                dynamic_cast<crossentropyLayer*>(mlp_layer[mlp_layer.size() - 1])->_cross_entropy(
                        mlp_layer[mlp_layer.size() - 1]->_y.transpose(), input_label);
        double cross_entropy_loss = cross_entropy_matrix.sum()
                                    / cross_entropy_matrix.rows();
        LOG(INFO) << "[Epoch/Iter]: [" << epoch << "/" << i << "], Loss=" << cross_entropy_loss << std::endl;
        // bp更新参数
        mlp_backward_update(input_label);
    }
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mlpClassifier::get_batch_data(
        const Eigen::MatrixXd &input_data,
        const Eigen::MatrixXd &input_label,
        std::stack<long> &index_stack) {
    int batch_size = 0;
    if (static_cast<int>(index_stack.size()) < _batch_size) {
        batch_size = static_cast<int>(index_stack.size());
    } else {
        batch_size = _batch_size;
    }

    Eigen::MatrixXd ret_first(batch_size, input_data.cols());
    Eigen::MatrixXd ret_second(batch_size, input_label.cols());
    for (auto i = 0; i < batch_size; ++i) {
        ret_first.row(i) = input_data.row(index_stack.top());
        ret_second.row(i) = input_label.row(index_stack.top());
        index_stack.pop();
    }
    return std::make_pair(ret_first, ret_second);
}

Eigen::MatrixXd mlpClassifier::one_hot_encode(const Eigen::MatrixXd &input) {
    double max_label_flag = -1.0f;
    for (auto i = 0; i < input.rows(); ++i) {
        if (input(i, 0) > max_label_flag) {
            max_label_flag = input(i, 0);
        }
    }
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(input.rows(),
                                                static_cast<long>(max_label_flag + 1));
    for (auto row = 0; row < input.rows(); ++row) {
        ret(row, static_cast<long>(input(row, 0))) = 1;
    }
    return ret;
}

Eigen::MatrixXd mlpClassifier::mlp_forward(const Eigen::MatrixXd &X) {
    // 初始化mlp输入
    mlp_layer[0]->_x = X.transpose();
    for (size_t layer_index = 0; layer_index < mlp_layer.size() - 1; ++layer_index) {
        mlp_layer[layer_index]->forward();
        mlp_layer[layer_index + 1]->_x = mlp_layer[layer_index]->_y;
    }
    Eigen::MatrixXd mlp_logits = mlp_layer[mlp_layer.size() - 2]->_y;
    softmaxLayer softmax_layer(static_cast<int>(mlp_logits.cols()),
                               _class_nums,
                               static_cast<int>(mlp_logits.rows()));
    // softmax输出
    softmax_layer._x = mlp_logits.transpose();
    softmax_layer.forward();
    return softmax_layer._y;
}

void mlpClassifier::mlp_backward_update(const Eigen::MatrixXd &Y) {
    mlp_layer[mlp_layer.size() - 1]->_dy = Y.transpose();
    //　后向传播
    for (auto i = static_cast<int>(mlp_layer.size() - 1); i >= 0; i--) {

        mlp_layer[i]->resetGrads();
        mlp_layer[i]->backward();
        // 最后一层不需要传播
        if (i > 0) {
            mlp_layer[i - 1]->_dy = mlp_layer[i]->_dx;
        }
    }
    // 更新梯度
    for (auto &layer : mlp_layer) {
        auto layer_tmp = dynamic_cast<linearTransformLayer*>(layer);
        if (layer_tmp != nullptr) {
#ifdef DEBUG
            LOG(INFO) << "原始权重为: " << layer_tmp->get_weights() << std::endl;
#endif
        }
        layer->applyGrads(_lr);
        if (layer_tmp != nullptr) {
#ifdef DEBUG
            LOG(INFO) << "权重梯度为: " << layer_tmp->get_weights_grad() << std::endl;
            LOG(INFO) << "更新后的权重为: " << layer_tmp->get_weights() << std::endl;
#endif
        }
    }
}
