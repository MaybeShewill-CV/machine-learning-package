/************************************************
* Copyright 2014 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: mnistDataloder.cpp
* Date: 18-4-28 上午11:06
************************************************/

#include "mnistDataloder.h"

#include <fstream>

#include <glog/logging.h>

#include <file_system_processor.h>

std::vector<mnist_sample> mnist_dataloder::load_mnist(const std::string &mnist_file_path,
                                                      const std::string &mnist_label_path) {
    assert(FileSystemProcessor::is_file_exist(mnist_file_path));
    assert(FileSystemProcessor::is_file_exist(mnist_label_path));

    const size_t offset_bytes = 16;
    const size_t offset_bytes_lab = 8;
    const size_t w = 28;
    const size_t h = 28;

    std::vector<mnist_sample> out;
    char buffer[w * h];
    char buffer_lab;

    size_t allocs = 0;

    std::ifstream infile(mnist_file_path.c_str(), std::ios::in|std::ios::binary);
    std::ifstream labels_file(mnist_label_path.c_str(), std::ios::in | std::ios::binary);

    if (infile.is_open() && labels_file.is_open()) {

        LOG(INFO) << "开始从" << mnist_file_path << "加载mnist数据" << std::endl;

        infile.seekg(offset_bytes, std::ios::beg);
        labels_file.seekg(offset_bytes_lab, std::ios::beg);

        while (!infile.eof() && !labels_file.eof()) {

            infile.read(buffer, w * h);
            labels_file.read(&buffer_lab, 1);

            if (!infile.eof() && !labels_file.eof()) {

                Eigen::VectorXd temp(w * h);

                allocs++;

                if (allocs % 1000 == 0) {
                    fflush(stdout);
                }

                for (size_t i = 0; i < w * h; i++) {
                    temp(i) = (double)((uint8_t)buffer[i]) / 255.0f;
                }

                mnist_sample sample;
                sample.first = temp;
                sample.second = static_cast<double >(buffer_lab);
                out.push_back(sample);
            }

        }

        LOG(INFO) << "mnist数据加载完毕" << std::endl;
        infile.close();
        labels_file.close();

    } else {

        LOG(ERROR) << "无法打开mnist数据文件" << mnist_file_path
                   << "或mnist数据标签文件" << mnist_label_path << std::endl;
    }
    return out;
}
