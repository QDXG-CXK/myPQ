#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <type_traits>

// 模板函数读取二进制文件
template<typename T>
std::vector<T> readBinaryFile(const std::string& filePath, size_t fileSize) {
    static_assert(std::is_trivially_copyable<T>::value, "Data type must be trivially copyable");

    std::vector<T> data(fileSize / sizeof(T));

    // 打开文件
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return std::vector<T>();
    }

    // 读取文件内容
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    
    // 检查是否读取成功
    if (!file) {
        std::cerr << "读取文件失败或读取的字节数不正确。" << std::endl;
        return std::vector<T>();
    }

    // 关闭文件
    file.close();

    return data;
}

int main() {
    std::string filePath = "mycodes.bin";
    size_t fileSize = 1024; // 假设文件大小为1024字节，实际大小请替换为正确的值

    // 调用函数读取文件，将数据读取成uint8_t类型的向量
    std::vector<uint8_t> fileData = readBinaryFile<uint8_t>(filePath, fileSize);

    // 输出读取的数据（仅示例）
    for (size_t i = 0; i < fileData.size(); ++i) {
        std::cout << std::hex << static_cast<int>(fileData[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}
