/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>
#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"
#include <chrono>

#include <omp.h>

const int64_t N_BASE = 400000000;
const int64_t BATCH_SIZE = 10000;
const int64_t N_QUERY = 1000;
const int64_t K = 5000;
const int64_t PQDIM = 50;
const int64_t SUBDIM = 4;
const int64_t NBITS = 8;
constexpr int64_t DIM = PQDIM * SUBDIM;
constexpr int64_t SUBK = 1 << NBITS;
const size_t SIZEOFDATA = 1;

OperatorDesc CreateOpDesc()
{
    // define operator
    std::vector<int64_t> inputShape0{PQDIM, SUBK, SUBDIM};
    std::vector<int64_t> inputShape1{BATCH_SIZE, PQDIM};  // for one batch of base
    std::vector<int64_t> inputShape2{N_QUERY, DIM};
    std::vector<int64_t> inputShape3{1};
    std::vector<int64_t> inputShape4{1};
    std::vector<int64_t> inputShape5{1};
    std::vector<int64_t> inputShape6{1};
    std::vector<int64_t> inputShape7{1};
    std::vector<int64_t> inputShape8{1};
    std::vector<int64_t> outputShape0{N_QUERY, K};
    std::vector<int64_t> outputShape1{N_QUERY, K};
    std::string opType = "PqsearchCust";

    OperatorDesc opDesc(opType);
    opDesc.AddInputTensorDesc(ACL_FLOAT, inputShape0.size(), inputShape0.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_UINT8, inputShape1.size(), inputShape1.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_INT8, inputShape2.size(), inputShape2.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_INT32, 1, inputShape3.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_INT32, 1, inputShape4.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_INT32, 1, inputShape5.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_INT32, 1, inputShape6.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_INT32, 1, inputShape7.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_INT32, 1, inputShape8.data(), ACL_FORMAT_ND);

    opDesc.AddOutputTensorDesc(ACL_INT32, outputShape0.size(), outputShape0.data(), ACL_FORMAT_ND);
    opDesc.AddOutputTensorDesc(ACL_FLOAT, outputShape1.size(), outputShape1.data(), ACL_FORMAT_ND);

    // int ret = aclrtSetOpExecuteTimeOut(3000U);
    //  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetOpExecuteTimeOut failed. ERROR: %d\n", ret); return ret);
    return opDesc;
}

struct KnnResults {
    KnnResults(size_t nq, size_t k, size_t nshard)
    {
        nq_ = nq;
        k_ = k;
        nshard_ = nshard;
        all_distances_ = new float[nshard * nq * k];
        all_labels_ = new int32_t[nshard * nq * k];
    }
    ~KnnResults()
    {
        delete all_distances_;
        delete all_labels_;
    }

    size_t nq_;
    size_t k_;
    size_t nshard_;
    float *all_distances_;
    int32_t *all_labels_;
};

bool SetInputData(OpRunner &runner)
{
    for (size_t i = 0; i < runner.NumInputs(); ++i) {
        if (i == 1 || i == 5) {
            continue;
        }  // input1 and input5 will be set in SetInputCodes()
        size_t fileSize = 0;
        std::string filePath = "test_data/data/input_" + std::to_string(i) + ".bin";
        bool ret = ReadFile(filePath, fileSize, runner.GetInputBuffer<void>(i), runner.GetInputSize(i));
        if (!ret) {
            ERROR_LOG("Read input[%zu] failed", i);
            return false;
        }
        INFO_LOG("file Size is:%zu.", fileSize);
        INFO_LOG("Set input[%zu] from %s success.", i, filePath.c_str());
        INFO_LOG("Input[%zu]:", i);
        runner.PrintInput(i);
        std::cout << std::endl;
        std::vector<int64_t> inputShape = runner.GetInputShape(i);
        std::string shapeString;
        for (size_t j = 0; j < inputShape.size(); j++) {
            shapeString.append(std::to_string(inputShape[j]));
            shapeString.append(" ");
        }
        INFO_LOG("Input[%zu] shape is:%s", i, shapeString.c_str());
    }

    return true;
}

bool SetInputCodes(OpRunner &runner, int64_t part_i, int64_t batch_size)
{
    int32_t *nb_p = runner.GetInputBuffer<int32_t>(5);
    nb_p[0] = batch_size;  // TODO: what if batch_size != BATCH_SIZE? Now the shape of codes is still (BATCH_SIZE,pqdim)
    INFO_LOG("Input[5]:");
    runner.PrintInput(5);
    std::cout << std::endl;

    std::string filePath = "test_data/data/input_1.bin";
    int64_t start = part_i * BATCH_SIZE * PQDIM * SIZEOFDATA;
    size_t dataSize = runner.GetInputSize(1) * batch_size / BATCH_SIZE;
    bool ret =
        ReadPartialFile(filePath, start, runner.GetInputBuffer<void>(1), dataSize);  // TODO: padding for last batch
    if (!ret) {
        ERROR_LOG("Read input[1] part-%zu failed", part_i);
        return false;
    }
    INFO_LOG("Set input[1] from %s success.", filePath.c_str());
    INFO_LOG("Input[1]:");
    runner.PrintInput(1);
    std::cout << std::endl;
    std::vector<int64_t> inputShape = runner.GetInputShape(1);
    std::string shapeString;
    for (size_t j = 0; j < inputShape.size(); j++) {
        shapeString.append(std::to_string(inputShape[j]));
        shapeString.append(" ");
    }
    INFO_LOG("Input[1] shape is:%s", shapeString.c_str());

    return true;
}

bool MergeOutputData(KnnResults &knnResults, size_t part_i, int32_t *labels, float *distances)
{  // merge multi output (NOTE: output labels is relatively of each batch)
    struct Compare {
        bool operator()(const std::pair<int32_t, float> &a, const std::pair<int32_t, float> &b)
        {
            return a.second < b.second;  // bigger one
        }
    };

    size_t shift = N_QUERY * K;  // shift between two shards

#pragma omp parallel for
    for (int iq = 0; iq < N_QUERY; iq++) {
        std::priority_queue<std::pair<int32_t, float>, std::vector<std::pair<int32_t, float>>, Compare> heap;

        for (int k = 0; k < K; k++) {  // batch_id == 0
            heap.emplace(knnResults.all_labels_[iq * K + k], knnResults.all_distances_[iq * K + k]);
        }
        
        for (int k = 0; k < K; k++) { // batch_id == part_i
            size_t id = shift + iq * K + k;
            if (knnResults.all_distances_[id] < heap.top().second) {
                heap.pop();
                heap.emplace(knnResults.all_labels_[id] + part_i * BATCH_SIZE, knnResults.all_distances_[id]);
            }
        }

        std::vector<std::pair<int32_t, float>> result;  // TODO: = Container(heap);
        while (!heap.empty()) {
            result.push_back(heap.top());
            heap.pop();
        }

        std::sort(result.begin(),
            result.end(),
            [](const std::pair<int32_t, float> &a, const std::pair<int32_t, float> &b) { return a.second < b.second; });

        for (int k = 0; k < K; k++) {
            size_t id = iq * K + k;
            labels[id] = result[k].first;
            distances[id] = result[k].second;
        }
    }

    return true;
}

bool SaveOutputData(OpRunner &runner, KnnResults &knnResults, size_t part_i)
{
    for (size_t i = 0; i < runner.NumOutputs(); ++i) {
        int64_t start;
        size_t dataSize;
        const void *src = runner.GetOutputBuffer<void>(i);
        void *dst;

        if (i == 0) {  // labels
            dataSize = N_QUERY * K * sizeof(knnResults.all_labels_[0]);
            start = (part_i == 0) ? 0 : dataSize;
            dst = static_cast<void *>(knnResults.all_labels_) + start;
        } else if (i == 1) {  // distances
            dataSize = N_QUERY * K * sizeof(knnResults.all_distances_[0]);
            start = (part_i == 0) ? 0 : dataSize;
            dst = static_cast<void *>(knnResults.all_distances_) + start;
        } else {
            ERROR_LOG("only 2 output.");
            return false;
        }

        memcpy(dst, src, dataSize);
    }

    if(part_i != 0){
        MergeOutputData(knnResults, part_i, knnResults.all_labels_, knnResults.all_distances_);
    }

    return true;
}

bool ProcessOutputData(OpRunner &runner, KnnResults &knnResults)
{
    int32_t *labels = knnResults.all_labels_;  // output0
    float *distances = knnResults.all_distances_;   // output1

    // print labels
    std::cout << "labels: (TOP5)";
    for (int i = 0; i < K * 5; i++) {
        if (i % 5 == 0) {
            std::cout << std::endl;
        }
        std::cout << labels[i] << "  ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < runner.NumOutputs(); ++i) {
        INFO_LOG("Output[%zu]:", i);
        // runner.PrintOutput(i);//means nothing
        std::cout << std::endl;
        std::string shapeString;
        std::vector<int64_t> outputShape = runner.GetOutputShape(i);
        for (size_t j = 0; j < outputShape.size(); j++) {
            shapeString.append(std::to_string(outputShape[j]));
            shapeString.append(" ");
        }
        INFO_LOG("Output[%zu] shape is:%s", i, shapeString.c_str());

        std::string filePath = "result_files/output_" + std::to_string(i) + ".bin";
        void *output;
        if (i == 0) {
            output = static_cast<void *>(labels);
        } else if (i == 1) {
            output = static_cast<void *>(distances);
        } else {
            ERROR_LOG("only 2 output.");
            return false;
        }

        if (!WriteFile(filePath, output, runner.GetOutputSize(i))) {
            ERROR_LOG("Write output[%zu] failed.", i);
            return false;
        }

        INFO_LOG("Write output[%zu] success. output file = %s", i, filePath.c_str());
    }

    return true;
}

bool RunPqsearchOp(bool isDevice)
{
    // Create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // Create Runner
    OpRunner opRunner(&opDesc, isDevice);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs except input1
    if (!SetInputData(opRunner)) {
        return false;
    }

    // Run op
    constexpr size_t nt = (N_BASE + BATCH_SIZE - 1) / BATCH_SIZE;
    KnnResults knnResults(N_QUERY, K, 2);
    for (size_t i = 0; i < nt; i++) {
        int64_t cur_batch_size = (i == nt - 1) ? N_BASE - BATCH_SIZE * (nt - 1) : BATCH_SIZE;
        INFO_LOG("%zu data will be processed in this batch", cur_batch_size);
        SetInputCodes(opRunner, i, cur_batch_size);
        if (!opRunner.RunOp()) {
            return false;
        }
        if (!SaveOutputData(opRunner, knnResults, i)) {
            return false;
        }
        INFO_LOG("---------------------------------- Process %zu/%zu success. ----------------------------------", i + 1, nt);
    }

    // Process output data
    if (!ProcessOutputData(opRunner, knnResults)) {
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main()
{
    // create result files path
    std::string output = "./result_files";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return FAILED;
        }
    }

    // init acl json
    if (aclInit("test_data/config/acl.json") != ACL_SUCCESS) {
        ERROR_LOG("Init acl failed");
        return FAILED;
    }

    // set model path
    int deviceId = 0;
    if (aclopSetModelDir("op_models") != ACL_SUCCESS) {
        std::cerr << "Load single op model failed" << std::endl;
        (void)aclFinalize();
        return FAILED;
    }

    // set device id
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::cerr << "Open device failed. device id = " << deviceId << std::endl;
        (void)aclFinalize();
        return FAILED;
    }
    INFO_LOG("Open device[%d] success", deviceId);

    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Acl get run mode failed");
        (void)aclrtResetDevice(deviceId);
        (void)aclFinalize();
        return FAILED;
    }
    bool isDevice = (runMode == ACL_DEVICE);

    auto start = std::chrono::high_resolution_clock::now();
    if (!RunPqsearchOp(isDevice)) {
        (void)aclrtResetDevice(deviceId);
        (void)aclFinalize();
        return FAILED;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> d1 = end - start;
    std::cout << "---op run " << d1.count() << " seconds ---" << std::endl;

    (void)aclrtResetDevice(deviceId);

    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        return FAILED;
    }

    return SUCCESS;
}
