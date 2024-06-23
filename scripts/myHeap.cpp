#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>

// 定义一个比较函数，用于最大堆
struct Compare {
    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first; // 最大的在前
    }
};

// 返回距离最小的K个点的距离和ID
std::vector<std::pair<float, int>> findClosestKPoints(const std::vector<float>& distances, const std::vector<int>& labels, int K) {
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, Compare> pq;

    for (size_t i = 0; i < distances.size(); ++i) {
        if(pq.size() < K) {
            pq.emplace(distances[i], labels[i]);//记得改为labels[i] + batch_id * BATCH_SIZE
        }
        else if(pq.top().first > distances[i]) {
            pq.pop();
            pq.emplace(distances[i], labels[i]);//记得改为labels[i] + batch_id * BATCH_SIZE
        }
    }

    std::vector<std::pair<float, int>> result;
    while (!pq.empty()) {
        result.push_back(pq.top());
        pq.pop();
    }

    std::sort(result.begin(), result.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    });
    //排完序再分别写入数组

    return result;
}

int main() {
    std::vector<float> distances = {1.2, 3.4, 0.5, 2.1, 0.9, 0.1, 0.2, 0.3, 5, 6, 7};
    std::vector<int> labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int K = 10;

    std::vector<std::pair<float, int>> closestPoints = findClosestKPoints(distances, labels, K);

    for (const auto& point : closestPoints) {
        std::cout << "Distance: " << point.first << ", ID: " << point.second << std::endl;
    }

    return 0;
}
