/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: globalutils.cpp
* Date: 18-4-9 下午5:36
************************************************/

#include "globalUtils.h"

#include <algorithm>

template <typename T>
std::vector<std::size_t> GlobalUtils::sort_indexes(const std::vector<T> &v) {
    // initialize original index locations
    std::vector<std::size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}