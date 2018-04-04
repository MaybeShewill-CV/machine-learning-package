// Copyright 2014 Baidu Inc. All Rights Reserved.
// Author: Luo Yao (luoyao@baidu.com)
// File: main.cpp
// Date: 18-4-3 下午2:29

#include <dataloder.h>
#include <LinearRegressionTrainer.h>

#include <glog/logging.h>

//#define TEST
//#define DATALOADER_TEST
#define LINEARREGRESSION_TEST

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetLogDestination(google::GLOG_INFO, "./log/ml_package_");
    google::SetStderrLogging(google::GLOG_INFO);

#ifdef DATALOADER_TEST
    if (argc != 2) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./dataloder txt文件路径" << std::endl;
        return -1;
    }
    DataLoder dataLoder;
    Eigen::MatrixXd datamatrix;
    dataLoder.load_data_from_txt(argv[1], datamatrix);
    LOG(INFO) << datamatrix(3, 2) << std::endl;
#endif

#ifdef LINEARREGRESSION_TEST
    if (argc != 2) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./linearregression txt文件路径" << std::endl;
        return -1;
    }
    LinearRegressionTrainer trainer;
    trainer.train(argv[1]);
#endif

#ifdef TEST
    Eigen::MatrixXd matrix1(2, 2);
    matrix1 << 1, 2, 3, 4;
    Eigen::MatrixXd matrix2(2, 2);
    matrix2 << 5, 6, 7, 8;
    LOG(INFO) << matrix1 << matrix2 << std::endl;

    LOG(INFO) << "Matrix1 col 1: " << matrix1.col(0) << std::endl;
    LOG(INFO) << "Matrix2 col 1: " << matrix2.col(0) << std::endl;
    auto dot_ret = matrix1.col(0).dot(matrix2.col(0));
    auto add_ret = matrix1.array() + 1;
    auto sum_ret = matrix1.sum();
    LOG(INFO) << "Add result is " << add_ret << std::endl;
    LOG(INFO) << "Dot result is " << dot_ret << std::endl;
    LOG(INFO) << "Sum result is " << sum_ret << std::endl;
#endif

    google::ShutdownGoogleLogging();

    return 0;
}


