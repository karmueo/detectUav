/**
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File utils.h
* Description: handle file operations
*/
#ifndef ACLLITE_TYPE_H
#define ACLLITE_TYPE_H
#pragma once

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include <memory>
#include <string>
#include <unistd.h>

enum MemoryType
{
    MEMORY_NORMAL = 0,
    MEMORY_HOST,
    MEMORY_DEVICE,
    MEMORY_DVPP,
    MEMORY_INVALID_TYPE
};

enum CopyDirection
{
    TO_DEVICE = 0,
    TO_HOST,
    INVALID_COPY_DIRECT
};

enum CameraId
{
    CAMERA_ID_0 = 0,
    CAMERA_ID_1,
    CAMERA_ID_INVALID,
};

enum VencStatus
{
    STATUS_VENC_INIT = 0,
    STATUS_VENC_WORK,
    STATUS_VENC_FINISH,
    STATUS_VENC_EXIT,
    STATUS_VENC_ERROR
};

// 编码数据回调函数类型
typedef void (*VencDataCallback)(void* data, uint32_t size, void* userData);

struct VencConfig
{
    uint32_t            maxWidth = 0;
    uint32_t            maxHeight = 0;
    std::string         outFile;
    acldvppPixelFormat  format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    acldvppStreamFormat enType = H264_MAIN_LEVEL;
    aclrtContext        context = nullptr;
    aclrtRunMode        runMode = ACL_HOST;
    VencDataCallback    dataCallback = nullptr;  // 编码数据回调
    void*               callbackUserData = nullptr; // 回调用户数据
    
    // H264 编码参数
    uint32_t            gopSize = 16;           // GOP大小(关键帧间隔),默认16
    uint32_t            rcMode = 2;             // 码率控制模式: 0=CBR, 1=VBR, 2=AVBR,默认2(AVBR)
    uint32_t            maxBitrate = 10000;     // 最大码率(kbps),默认10000 (10Mbps)
    
    // RTSP输出参数
    uint32_t            outputWidth = 1920;     // 输出视频宽度,默认1920
    uint32_t            outputHeight = 1080;    // 输出视频高度,默认1080
    uint32_t            outputFps = 25;         // 输出视频帧率,默认25
    std::string         rtspTransport = "tcp";  // RTSP传输协议("tcp"或"udp"),默认"tcp"
    uint32_t            rtspBufferSize = 1024000;  // RTSP缓冲区大小(字节),默认1024000 (1MB)
    uint32_t            rtspMaxDelay = 500000;  // RTSP最大延迟(微秒),默认500000 (0.5s)
};

struct ImageData
{
    acldvppPixelFormat       format;
    uint32_t                 width = 0;
    uint32_t                 height = 0;
    uint32_t                 alignWidth = 0;
    uint32_t                 alignHeight = 0;
    uint32_t                 size = 0;
    std::shared_ptr<uint8_t> data = nullptr;
};

struct FrameData
{
    bool     isFinished = false;
    uint32_t frameId = 0;
    uint32_t size = 0;
    void    *data = nullptr;
};

struct Resolution
{
    uint32_t width = 0;
    uint32_t height = 0;
};

struct Rect
{
    uint32_t ltX = 0;
    uint32_t ltY = 0;
    uint32_t rbX = 0;
    uint32_t rbY = 0;
};

struct BBox
{
    Rect        rect;
    uint32_t    score = 0;
    std::string text;
};

struct AclLiteMessage
{
    int                   dest;
    int                   msgId;
    std::shared_ptr<void> data = nullptr;
};

struct DataInfo
{
    void    *data;
    uint32_t size;
};

struct InferenceOutput
{
    std::shared_ptr<void> data = nullptr;
    uint32_t              size;
};

struct ModelOutputInfo
{
    const char  *name;
    aclmdlIODims dims;
    aclFormat    format;
    aclDataType  dataType;
};

#endif