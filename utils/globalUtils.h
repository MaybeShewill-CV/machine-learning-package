/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: globalutils.h
* Date: 18-4-9 下午5:36
************************************************/

#ifndef MACHINE_LEARNING_PACKAGE_GLOBALUTILS_H
#define MACHINE_LEARNING_PACKAGE_GLOBALUTILS_H

#include <vector>
#include <map>
#include <set>
#include <algorithm>

class GlobalUtils {
public:
    GlobalUtils() = default;
    ~GlobalUtils() = default;

    template <typename T>
    static std::vector<std::size_t> sort_indexes(const std::vector<T> &v) {
        std::vector<std::size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(),
                  [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

        return idx;
    };

    template <typename T, typename V>
    static bool has_key(const std::map<T, V> &kv_map, const T& key) {
        return kv_map.count(key) > 0;
    };

    template <typename T>
    static bool has_elements(const std::set<T> &set_tmp, const T &ele_val) {
        return set_tmp.count(ele_val) > 0;
    }

    template <typename T>
    static bool has_elements(const std::vector<T> &vec, const T &ele_val) {
        for (auto &ele : vec) {
            if (ele == ele_val) {
                return true;
            }
        }
        return false;
    }
};

#endif //MACHINE_LEARNING_PACKAGE_GLOBALUTILS_H
