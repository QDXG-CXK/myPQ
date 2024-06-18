#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// 模板函数读取二进制文件
template<typename T>
std::vector<T> readBinaryFile(const std::string& filePath, size_t fileSize) {

    std::vector<T> data(fileSize / sizeof(T));

    // 打开文件
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "can't open file " << filePath << std::endl;
        return std::vector<T>();
    }

    // 读取文件内容
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    
    // 检查是否读取成功
    if (!file) {
        std::cerr << "failure." << std::endl;
        return std::vector<T>();
    }

    // 关闭文件
    file.close();

    return data;
}

int main() {
    // 调用函数读取文件，将数据读取成uint8_t类型的向量
    std::vector<int64_t> fileData = readBinaryFile<int64_t>("/home/algo/xdu/normal_cpu/myPQ/data/float/gt_I.bin", 4000);
    // 输出读取的数据（仅示例）
    for (size_t i = 0; i < fileData.size(); ++i) {
        if(i%5==0) std::cout<<std::endl;
        std::cout << fileData[i] << " ";
    }
    std::cout << std::endl;

    // 调用函数读取文件，将数据读取成uint8_t类型的向量
    std::vector<float> fileData2 = readBinaryFile<float>("/home/algo/xdu/normal_cpu/myPQ/data/float/gt_D.bin", 2000);
    // 输出读取的数据（仅示例）
    for (size_t i = 0; i < fileData2.size(); ++i) {
        if(i%5==0) std::cout<<std::endl;
        std::cout << fileData2[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
