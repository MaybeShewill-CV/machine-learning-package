/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: dataloder.cpp
* Date: 18-4-3 下午4:15
************************************************/

#include "dataloder.h"

#include <fstream>

#include <file_system_processor.h>
#include <boost/algorithm/string.hpp>

DataLoder::DataLoder() = default;

DataLoder::~DataLoder() = default;

void DataLoder::load_data_from_txt(const std::string &txt_path,
                                   Eigen::MatrixXd &data_matrix) {
    assert(FileSystemProcessor::is_file_exist(txt_path));

    std::fstream input_file(txt_path, std::ios_base::in);
    assert(input_file.is_open());

    char buffer[256];
    std::vector<std::vector<double> > data_vec;
    while (!input_file.eof()) {
        input_file.getline(buffer, 1000);
        std::stringstream inputstream(buffer);
        double feats = 0.0;
        if (strcmp(buffer, "\n") != 0) {
            std::vector<double> tmp;
            while (inputstream >> feats) {
                tmp.push_back(feats);
            }
            data_vec.push_back(tmp);
        }
    }
    data_vec.pop_back();

    data_matrix = Eigen::MatrixXd::Random(data_vec.size(), data_vec[0].size());
    int row = 0;
    for (auto &feats_vec : data_vec) {
        int col = 0;
        for (auto &feats : feats_vec) {
            data_matrix(row, col) = feats;
            col++;
        }
        row++;
    }
}