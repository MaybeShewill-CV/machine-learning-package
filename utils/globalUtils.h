/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: globalutils.h
* Date: 18-4-9 下午5:36
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_GLOBALUTILS_H
#define MACHINE_LEARNING_PACKAGE_GLOBALUTILS_H

#include <vector>


class GlobalUtils {
public:
    GlobalUtils() = default;
    ~GlobalUtils() = default;

    template <typename T>
    static std::vector<std::size_t> sort_indexes(const std::vector<T> &v);
};

#endif //MACHINE_LEARNING_PACKAGE_GLOBALUTILS_H
