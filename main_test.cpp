// Date: 18-4-11 下午4:25
// Copyright 2014 Baidu Inc. All Rights Reserved.
// Author: Luo Yao (luoyao@baidu.com)
// File: main.cpp
// Date: 18-4-3 下午2:29

#include <dataloder.h>
#include <linearRegression.h>
#include <knnClassiferTrainer.h>
#include <logisticClassifierTrainer.h>
#include <decisionTreeClassiferTrainer.h>
#include <kmeansClusterTrainer.h>
#include <lvqClusterTrainer.h>
#include <gmmClusterTrainer.h>
#include <dbscanClusterTrainer.h>
#include <mlpClassifierTrainer.h>

//#define TEST
//#define DATALOADER_TEST
//#define LINEARREGRESSION_TEST
//#define KNNCLASSIFIER_TEST
//#define LOGISTICCLASSIFIER_TEST
//#define DECISIONTREE_TEST
//#define KMEANS_TEST
//#define LVQ_TEST
//#define GMM_TEST
//#define DBSCAN_TEST
#define MLPCLASSIFIER_TEST

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
    trainer.test(argv[1]);
    trainer.deploy(argv[1]);
#endif

#ifdef KNNCLASSIFIER_TEST
    if (argc != 4) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./knnclassifier knn训练数据 knn测试数据 knn验证数据" << std::endl;
        return -1;
    }
    knnClassiferTrainer trainer;

    trainer.train(argv[1]);
    trainer.test(argv[2]);
    trainer.deploy(argv[3]);
#endif

#ifdef LOGISTICCLASSIFIER_TEST
    if (argc != 4) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./logisticclassifier logistic训练数据 logistic测试数据 logistic验证数据" << std::endl;
        return -1;
    }
    logisticClassifierTrainer trainer(0.1, 5000);

    trainer.train(argv[1]);
    trainer.test(argv[2]);
    trainer.deploy(argv[3]);
#endif

#ifdef DECISIONTREE_TEST
    if (argc != 4) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./decisiontreeclassifier dtree训练数据 dtree测试数据 dtree验证数据" << std::endl;
        return -1;
    }
    decisionTreeClassiferTrainer trainer(ID3_DTREE);
    trainer.train(argv[1]);
    trainer.test(argv[2]);
    trainer.deploy(argv[3]);
#endif

#ifdef KMEANS_TEST
    if (argc != 4) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./kmeansCluster kmeans训练数据 kmeans测试数据 kmeans验证数据" << std::endl;
        return -1;
    }
    kmeansClusterTrainer clusterTrainer(5, 1000, EUCLIDEAN);

    clusterTrainer.train(argv[1]);
    clusterTrainer.test(argv[2]);
    clusterTrainer.deploy(argv[3]);
#endif

#ifdef LVQ_TEST
    if (argc != 4) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./lvqCluster lvq训练数据 lvq测试数据 lvq验证数据" << std::endl;
        return -1;
    }
    lvqClusterTrainer clusterTrainer(5, 1000, 0.1);
    clusterTrainer.train(argv[1]);
    clusterTrainer.test(argv[2]);
    clusterTrainer.deploy(argv[3]);
#endif

#ifdef GMM_TEST
    if (argc != 4) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./gmmCluster gmm训练数据 gmm测试数据 gmm验证数据" << std::endl;
        return -1;
    }
    gmmClusterTrainer clusterTrainer(5, 250);
    clusterTrainer.train(argv[1]);
    clusterTrainer.test(argv[2]);
    clusterTrainer.deploy(argv[3]);
#endif

#ifdef DBSCAN_TEST
    if (argc != 4) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./dbscanCluster dbscan训练数据 dbscan测试数据 dbscan验证数据" << std::endl;
        return -1;
    }
    dbscanClusterTrainer clusterTrainer(5, 0.11);
    clusterTrainer.train(argv[1]);
    clusterTrainer.test(argv[2]);
    clusterTrainer.deploy(argv[3]);
#endif

#ifdef MLPCLASSIFIER_TEST
    if (argc != 4) {
        LOG(INFO) << "Usage: " << std::endl;
        LOG(INFO) << "./mlpClassifier mlp训练数据 mlp测试数据 mlp验证数据" << std::endl;
        return -1;
    }
    mlpClassifierTrainer trainer(10, 0.001, 5, 250, 28 * 28);
    trainer.train(argv[1]);
    trainer.test(argv[2]);
#endif

#ifdef TEST
    Eigen::MatrixXd matrix1(1, 2);
    matrix1 << 1, 2;
    Eigen::MatrixXd matrix2(2, 1);
    matrix2 << 5, 6;
    LOG(INFO) << matrix1 * matrix2 << std::endl;
    Eigen::RowVectorXd rowvector(4);
    rowvector << 1, 2, 3, 4;
    Eigen::RowVectorXd rowvector2(4);
    rowvector2 << 2, 3, 4, 5;
    LOG(INFO) << "Rowvector diff is " << rowvector2 - rowvector << std::endl;

    LOG(INFO) << "Matrix1 col 1: " << matrix1.col(0) << std::endl;
    LOG(INFO) << "Matrix2 col 1: " << matrix2.col(0) << std::endl;
    auto dot_ret = matrix1.col(0).dot(matrix2.col(0));
    auto add_ret = matrix1.array() + 1;
    auto sum_ret = matrix1.sum();
    Eigen::RowVectorXd multiply_ret = matrix1 * rowvector.transpose();
    LOG(INFO) << "Add result is " << add_ret << std::endl;
    LOG(INFO) << "Dot result is " << dot_ret << std::endl;
    LOG(INFO) << "Sum result is " << sum_ret << std::endl;
    LOG(INFO) << "Multiply result is " << multiply_ret <<  std::endl;
#endif

     google::ShutdownGoogleLogging();

    return 0;
}

