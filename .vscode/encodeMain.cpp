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
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <stdint.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"
#include <chrono>

const int64_t N_BASE = 400000000;
const int64_t BATCH_SIZE = 100000;
const int64_t NBITS = 8;
const int64_t PQDIM = 50;
const int64_t SUBDIM = 4;
constexpr int64_t DIM = PQDIM * SUBDIM;
const size_t SIZEOFDATA = 1;
constexpr int64_t SUBK = 1 << NBITS;
uint8_t *CODES;

OperatorDesc CreateOpDesc()
{
    // define operator
    std::vector<int64_t> inputShape0{PQDIM, SUBK, SUBDIM};
    std::vector<int64_t> inputShape1{1};
    std::vector<int64_t> inputShape2{1};
    std::vector<int64_t> inputShape3{1};
    std::vector<int64_t> inputShape4{BATCH_SIZE, DIM};
    std::vector<int64_t> inputShape5{1};
    std::vector<int64_t> inputShape6{1};

    std::vector<int64_t> outputShape{BATCH_SIZE, PQDIM};
    std::string opType = "PqencodeCust";

    aclFormat format = ACL_FORMAT_ND;
    aclDataType dataType = ACL_INT32;
    OperatorDesc opDesc(opType);
    opDesc.AddInputTensorDesc(ACL_FLOAT, inputShape0.size(), inputShape0.data(), format);
    opDesc.AddInputTensorDesc(dataType, 1, inputShape1.data(), format);
    opDesc.AddInputTensorDesc(dataType, 1, inputShape2.data(), format);
    opDesc.AddInputTensorDesc(dataType, 1, inputShape3.data(), format);
    opDesc.AddInputTensorDesc(ACL_INT8, inputShape4.size(), inputShape4.data(), format);
    opDesc.AddInputTensorDesc(dataType, 1, inputShape5.data(), format);
    opDesc.AddInputTensorDesc(dataType, 1, inputShape6.data(), format);
    opDesc.AddOutputTensorDesc(ACL_UINT8, outputShape.size(), outputShape.data(), format);

    //int ret = aclrtSetOpExecuteTimeOut(3000U);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetOpExecuteTimeOut failed. ERROR: %d\n", ret); return ret);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    for (size_t i = 0; i < runner.NumInputs(); ++i) {
        if(i == 4){//xb
            continue;
        }
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

bool SetInputBase(OpRunner &runner, int64_t part_i, int64_t batch_size)
{
    std::string filePath = "test_data/data/input_4.bin";
    int64_t start = part_i * BATCH_SIZE * DIM * SIZEOFDATA;
    size_t dataSize = runner.GetInputSize(4) * batch_size / BATCH_SIZE;
    bool ret = ReadPartialFile(filePath, start, runner.GetInputBuffer<void>(4), dataSize);
    if (!ret) {
            ERROR_LOG("Read input[4] part-%zu failed", part_i);
            return false;
    }
    INFO_LOG("Set input[4] from %s success.", filePath.c_str());
    INFO_LOG("Input[4]:");
    runner.PrintInput(4);
    std::cout << std::endl;
    std::vector<int64_t> inputShape = runner.GetInputShape(4);
    std::string shapeString;
    for (size_t j = 0; j < inputShape.size(); j++) {
        shapeString.append(std::to_string(inputShape[j]));
        shapeString.append(" ");
    }
    INFO_LOG("Input[4] shape is:%s", shapeString.c_str());

    return true;
}

bool SaveOutputData(OpRunner &runner, size_t part_i, int64_t batch_size)
{
    const size_t dataSize = batch_size * PQDIM * sizeof(uint8_t);//Warning: only support nbits == 8 (TODO: stochastic nbits)
    const size_t start = part_i * BATCH_SIZE * PQDIM * sizeof(uint8_t);
    const void* src = runner.GetOutputBuffer<void>(0);
    void* dst = static_cast<void*>(CODES) + start;

    if(start + dataSize > N_BASE * PQDIM * sizeof(uint8_t)){
        ERROR_LOG("CODES: out of ranges.");
        return false;
    }

    memcpy(dst, src, dataSize);

    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    for (size_t i = 0; i < runner.NumOutputs(); ++i) {
        if (i > 0){
            ERROR_LOG("Only 1 output.");
            return false;
        }
        INFO_LOG("Output[%zu]: (only display TOP-5 and LAST-5)", i);
        for(int i = 0; i < 5; i++){
            for(int j = 0; j < PQDIM; j++){
                std::cout<<int(CODES[i * PQDIM + j])<<"  ";
            }
            std::cout<<std::endl;
        }
        std::cout<<"\n\t\t.\n\t\t.\n\t\t.\n\t\t.\n\t\t.\n\t\t.\n\n";
        for(int i = N_BASE-5; i < N_BASE; i++){
            for(int j = 0; j < PQDIM; j++){
                std::cout<<int(CODES[i * PQDIM + j])<<"  ";
            }
            std::cout<<std::endl;
        }

        std::string filePath = "result_files/output_" + std::to_string(i) + ".bin";
        size_t fileSize = N_BASE * PQDIM * sizeof(uint8_t);//Warning: only support nbits == 8 (TODO: stochastic nbits)
        if (!WriteFile(filePath, static_cast<void*>(CODES), fileSize)) {
            ERROR_LOG("Write output[%zu] failed.", i);
            return false;
        }

        INFO_LOG("Write output[%zu] success. output file = %s", i, filePath.c_str());
    }
    return true;
}

bool RunPqencodeOp(bool isDevice)
{
    // Create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // Create Runner
    OpRunner opRunner(&opDesc, isDevice);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        return false;
    }

    // Run op
    constexpr size_t nt = (N_BASE + BATCH_SIZE -1) / BATCH_SIZE;
    CODES = new uint8_t[N_BASE * PQDIM];//Warning: only support nbits == 8 (TODO: stochastic nbits)
    for(size_t i=0; i<nt; i++){
        int64_t cur_batch_size = (i == nt-1) ? N_BASE - BATCH_SIZE * (nt-1) : BATCH_SIZE;
        SetInputBase(opRunner, i, cur_batch_size);
        if (!opRunner.RunOp()) {
            return false;
        }
        if (!SaveOutputData(opRunner, i, cur_batch_size)) {
            return false;
        }
        INFO_LOG("Process %zu/%zu success.", i+1, nt);
    }

    // Process output data
    if (!ProcessOutputData(opRunner)) {//TODO: ProcessOutputData, save all codes
        return false;
    }

    INFO_LOG("Run op success");
    delete CODES;
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
    if (!RunPqencodeOp(isDevice)) {
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
