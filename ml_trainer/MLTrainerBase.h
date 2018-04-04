/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: mLTrainerBase.h
* Date: 18-4-4 下午2:45
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_MLTRAINERBASE_H
#define MACHINE_LEARNING_PACKAGE_MLTRAINERBASE_H

#include <string>
#include <dataloder.h>

class MLTrainerBase {
public:
    MLTrainerBase()= default;
    virtual ~MLTrainerBase() = default;

    virtual void train(const std::string& input_file_path) = 0;

private:
    DataLoder dataLoder;

};


#endif //MACHINE_LEARNING_PACKAGE_MLTRAINERBASE_H
