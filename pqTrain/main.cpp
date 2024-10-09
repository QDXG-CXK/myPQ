#include <string>
#include <cmath>
#include <string.h>
#include <iostream>
#include <fstream>
#include "clustering.h"
#include "omp.h"
#include <chrono>

// 采用CPP模式读二进制文件
void BinReadInt8(std::string path, int8_t *des, size_t data_num)
{

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cout << "读取文件失败" << std::endl;
        return;
    }
    f.read((char *)des, data_num * sizeof(int8_t));
    f.close();
}

void BinWriteFloat(std::string path, float *des, size_t data_num)
{

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cout << "打开文件失败" << std::endl;
        return;
    }
    f.write((char *)des, data_num * sizeof(float));
    f.close();
}

int main()
{
    std::string read_path = "/data/data02/dataset/Text2Image_256d_600M/samples_100_2_1M.bin";
    std::string save_path = "/data/data02/dataset/Text2Image_256d_600M/codebook_100_2_1M.bin";
    uint32_t ns = 1000000;
    uint32_t pqdim = 100;
    uint32_t subdim = 2;
    uint32_t nbits = 8;
    uint32_t nbits_2pow = 1 << nbits;

    int64_t total = pqdim;
    int64_t startpos = 0;
    int64_t len = total;
    // uint32_t per_unit = total;

    size_t data_num = pqdim * nbits_2pow * subdim;
    float* codebook = new float[data_num];

    int64_t blockdim = 10;  //对应AI CPU的数目 用于平摊任务

    std::vector<int8_t> xs(ns * pqdim * subdim, 0);
    BinReadInt8(read_path, xs.data(), ns * pqdim * subdim);
    auto start = std::chrono::high_resolution_clock::now();
    for (int blockid = 0; blockid < blockdim; blockid++) {
        uint32_t per_unit = std::ceil(total / blockdim);
        uint32_t startpos = blockid * per_unit;
        uint32_t len = blockid < blockdim - 1 ? per_unit : (total - per_unit * (blockdim - 1));
        std::cout << "blockid:" << blockid << " per_unit:" << per_unit << " startpos:" << startpos << " len:" << len
                  << std::endl;
    }

#pragma omp parallel for
    for (int blockid = 0; blockid < blockdim; blockid++) {

        uint32_t per_unit = std::ceil(total / blockdim);
        uint32_t startpos = blockid * per_unit;
        uint32_t len = blockid < blockdim - 1 ? per_unit : (total - per_unit * (blockdim - 1));
        for (uint32_t ith_pqdim = startpos; ith_pqdim < startpos + len && ith_pqdim < total; ith_pqdim++) {
            Kmeans<int8_t> clus(subdim, nbits_2pow);
            std::vector<float> centroids = clus.train(ns, xs.data() + subdim * ns * ith_pqdim);
            std::cout << "ith_pqdim = " << ith_pqdim << "; centroids.size() = " << centroids.size() <<"\n";
            memcpy(codebook + ith_pqdim * nbits_2pow * subdim, centroids.data(), nbits_2pow * subdim * sizeof(float));
        }

        // for (int i = 0; i < nbits_2pow; i++) {
        //     for (int j = 0; j < subdim; j++) {
        //         std::cout << centroids[i * subdim + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> d1 = end - start;
    std::cout << "---op run " << d1.count() << " seconds ---" << std::endl;

    BinWriteFloat(save_path, codebook, data_num);

    delete[] codebook;
    // std::cout << "kmeans tests" << std::endl;
}
