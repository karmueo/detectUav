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
#ifndef DATAINPUTTHREAD_H
#define DATAINPUTTHREAD_H
#pragma once
#include "AclLiteApp.h"
#include "AclLiteImageProc.h"
#include "AclLiteThread.h"
#include "Params.h"
#include "VideoCapture.h"
#include <mutex>
#include <unistd.h>

class DataInputThread : public AclLiteThread
{
  public:
    DataInputThread(int32_t       deviceId,
                    int32_t       channelId,
                    aclrtRunMode &runMode,
                    std::string   inputDataType,
                    std::string   inputDataPath,
                    std::string   inferName,
                    int           postThreadNum,
                    uint32_t      batch,
                    int           framesPerSecond,
                    int           frameSkip = 0,
                    std::string   outputType = "");

    ~DataInputThread();
    AclLiteError Init();
    AclLiteError Process(int msgId, std::shared_ptr<void> msgData);

  private:
    AclLiteError AppStart();
    AclLiteError MsgRead(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    AclLiteError MsgSend(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    AclLiteError OpenPicsDir();
    AclLiteError OpenVideoCapture();
    AclLiteError ReadPic(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    AclLiteError ReadStream(std::shared_ptr<DetectDataMsg> &detectDataMsg);
    AclLiteError GetOneFrame(std::shared_ptr<DetectDataMsg> &detectDataMsg);

  private:
    uint32_t deviceId_;
    uint32_t channelId_;
    int      frameCnt_;
    int      msgNum_;
    uint32_t batch_;
    bool     readFrameReady_;  // flag for MSG_READ_FRAME received
    bool     inferDoneReady_;  // flag for MSG_INFER_DONE received

    std::string inputDataType_;
    std::string inputDataPath_;
    std::string inferName_;
    std::string outputType_;
    int         postThreadNum_;
    int         postproId_;

    aclrtRunMode      runMode_;
    AclLiteVideoProc *cap_;
    AclLiteImageProc  dvpp_;

    int                      selfThreadId_;
    int                      preThreadId_;
    int                      inferThreadId_;
    std::vector<int>         postThreadId_;
    int                      dataOutputThreadId_;
    int                      rtspDisplayThreadId_;
    int                      hdmiDisplayThreadId_;
    std::vector<std::string> fileVec_;

    int64_t lastDecodeTime_;
    int64_t realWaitTime_;
    int64_t waitTime_;
    int     framesPerSecond_;
    int     frameSkip_;  // 跳帧参数: 跳过 frameSkip_ 帧; 0 = 不跳帧 (process every frame)
    
    // ============ 跟踪状态管理 ============
    int     trackThreadId_;              // 跟踪线程id
    bool    isTrackingActive_;           // 当前跟踪是否活跃
    float   currentTrackingConfidence_;  // 当前跟踪置信度
    bool    isFirstFrame_;               // 是否是第一帧(首帧必须进行检测)
    int64_t lastTrackingLostTime_;       // 上次跟踪丢失时间戳(微秒), 用于防止重复处理
};

#endif
