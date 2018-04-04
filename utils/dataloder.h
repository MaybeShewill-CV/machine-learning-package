/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: dataloder.h
* Date: 18-4-3 下午4:15
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_DATALODER_H
#define MACHINE_LEARNING_PACKAGE_DATALODER_H

#include <string>
#include <eigen3/Eigen/Dense>

class DataLoder {
public:
    DataLoder();
    ~DataLoder();

    void load_data_from_txt(const std::string& txt_path, Eigen::MatrixXd& data_matrix);

};


#endif //MACHINE_LEARNING_PACKAGE_DATALODER_H
