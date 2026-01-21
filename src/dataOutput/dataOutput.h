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

* File sample_process.h
* Description: handle acl resource
*/
#ifndef DATAOUTPUTTHREAD_H
#define DATAOUTPUTTHREAD_H
#pragma once

#include "AclLiteApp.h"
#include "AclLiteError.h"
#include "AclLiteThread.h"
#include "AclLiteUtils.h"
#include "Params.h"
#include "acl/acl.h"
#include "AclLiteType.h"
#include "AclLiteImageProc.h"
#include <iostream>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unistd.h>

class DataOutputThread : public AclLiteThread
{
  public:
    DataOutputThread(aclrtRunMode &runMode,
             std::string   outputDataType,
             std::string   outputPath,
             int           postThreadNum,
             VencConfig    vencConfig = VencConfig());
    ~DataOutputThread();

    AclLiteError Init();
    AclLiteError Process(int msgId, std::shared_ptr<void> data);

  private:
    AclLiteError SetOutputVideo();
    AclLiteError ShutDownProcess();
    AclLiteError RecordQueue(std::shared_ptr<DetectDataMsg> detectDataMsg);
    AclLiteError DataProcess();
    AclLiteError ProcessOutput(std::shared_ptr<DetectDataMsg> detectDataMsg);

    AclLiteError SaveResultVideo(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    AclLiteError SaveResultPic(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    AclLiteError PrintResult(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    AclLiteError SendCVImshow(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    AclLiteError DisplayMsgSend(std::shared_ptr<DetectDataMsg> detectDataMsg);
    AclLiteError SendImageToRtsp(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    AclLiteError SendImageToHdmi(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    void         UpdateCachedResult(
                const std::shared_ptr<DetectDataMsg> &detectDataMsg);
    void ApplyCachedResult(std::shared_ptr<DetectDataMsg> &detectDataMsg);

  private:
    aclrtRunMode                               runMode_;
    cv::VideoWriter                            outputVideo_;
    std::string                                outputDataType_;
    std::string                                outputPath_;
    int                                        shutdown_;
    int                                        postNum_;
    std::queue<std::shared_ptr<DetectDataMsg>> postQueue_[4];
    uint32_t                                   frameCnt_;
    int64_t                                    lastDecodeTime_;
    int64_t                                    lastRecordTime_;
    VencConfig                                 g_vencConfig;
    AclLiteImageProc                            dvpp_;
    struct CachedResult
    {
        std::vector<DetectionOBB> detections;
        TrackInfo                 trackingResult;
        std::vector<std::string>  textPrint;
        bool                      trackingActive = false;
        float                     trackingConfidence = 0.0f;
        bool                      filterStaticTargetEnabled = false;
        bool                      hasBlockedTarget = false;
        float                     blockedCenterX = 0.0f;
        float                     blockedCenterY = 0.0f;
        float                     blockedWidth = 0.0f;
        float                     blockedHeight = 0.0f;
        float                     staticCenterThreshold = 0.0f;
        float                     staticSizeThreshold = 0.0f;
    };
    std::unordered_map<uint32_t, CachedResult> lastResults_;
    std::unordered_map<uint32_t, int>          lastOutputMsgNum_; // 每路通道最后输出的帧序号
};

#endif
