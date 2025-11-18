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

* File sample_process.cpp
* Description: handle acl resource
*/
#include "dataInput.h"
#include "Params.h"
#include <chrono>
#include <sys/time.h>

namespace
{
const uint32_t kYuvMultiplier = 3;
const uint32_t kYuvDivisor = 2;
const uint32_t kSleepTime = 500;
const uint32_t kOneSec = 1000000;
const uint32_t kOneMSec = 1000;
const uint32_t kTargetFPS = 25;  // 目标帧率,控制输入速度
} // namespace
using namespace std;

DataInputThread::DataInputThread(int32_t       deviceId,
                                 int32_t       channelId,
                                 aclrtRunMode &runMode,
                                 string        inputDataType,
                                 string        inputDataPath,
                                 string        inferName,
                                 int           postThreadNum,
                                 uint32_t      batch,
                                 int           framesPerSecond,
                                 int           frameSkip)
    : deviceId_(deviceId),
      channelId_(channelId),
      frameCnt_(0),
      msgNum_(0),
      batch_(batch),
      readFrameReady_(false),
      inferDoneReady_(true),
      inputDataType_(inputDataType),
      inputDataPath_(inputDataPath),
      inferName_(inferName),
      postThreadNum_(postThreadNum),
      postproId_(0),
      runMode_(runMode),
      cap_(nullptr),
      selfThreadId_(INVALID_INSTANCE_ID),
      preThreadId_(INVALID_INSTANCE_ID),
      inferThreadId_(INVALID_INSTANCE_ID),
      postThreadId_(postThreadNum, INVALID_INSTANCE_ID),
      dataOutputThreadId_(INVALID_INSTANCE_ID),
      rtspDisplayThreadId_(INVALID_INSTANCE_ID),
      framesPerSecond_(framesPerSecond),
      frameSkip_(frameSkip < 1 ? 1 : frameSkip)  // 至少为1,不跳帧
{
}

DataInputThread::~DataInputThread()
{
    if (inputDataType_ == "pic")
    {
        dvpp_.DestroyResource();
    }
    if (cap_ != nullptr)
    {
        cap_->Close();
        delete cap_;
        cap_ = nullptr;
    }
}

AclLiteError DataInputThread::OpenPicsDir()
{
    string inputImageDir = inputDataPath_;
    GetAllFiles(inputImageDir, fileVec_);
    if (fileVec_.empty())
    {
        ACLLITE_LOG_ERROR("Failed to deal all empty path=%s.",
                          inputImageDir.c_str());
        return ACLLITE_ERROR;
    }

    return ACLLITE_OK;
}

AclLiteError DataInputThread::OpenVideoCapture()
{
    if (IsRtspAddr(inputDataPath_))
    {
        cap_ = new AclLiteVideoProc(inputDataPath_, deviceId_);
    }
    else if (IsVideoFile(inputDataPath_))
    {
        if (!IsPathExist(inputDataPath_))
        {
            ACLLITE_LOG_ERROR("The %s is inaccessible", inputDataPath_.c_str());
            return ACLLITE_ERROR;
        }
        cap_ = new AclLiteVideoProc(inputDataPath_, deviceId_);
    }
    else
    {
        ACLLITE_LOG_ERROR("Invalid param. The arg should be accessible rtsp,"
                          " video file or camera id");
        return ACLLITE_ERROR;
    }

    if (!cap_->IsOpened())
    {
        delete cap_;
        cap_ = nullptr;
        ACLLITE_LOG_ERROR("Failed to open video");
        return ACLLITE_ERROR;
    }

    return ACLLITE_OK;
}

AclLiteError DataInputThread::Init()
{
    AclLiteError aclRet;
    if (inputDataType_ == "pic")
    {
        aclRet = OpenPicsDir();
        if (aclRet != ACLLITE_OK)
        {
            return ACLLITE_ERROR;
        }
        aclRet = dvpp_.Init("DVPP_CHNMODE_JPEGD");
        if (aclRet)
        {
            ACLLITE_LOG_ERROR("Dvpp init failed, error %d", aclRet);
            return ACLLITE_ERROR;
        }
    }
    else
    {
        aclRet = OpenVideoCapture();
        if (aclRet != ACLLITE_OK)
        {
            return ACLLITE_ERROR;
        }
    }
    // Get the relevant thread instance id
    // 获取相关线程实例id
    selfThreadId_ = SelfInstanceId();
    inferThreadId_ = GetAclLiteThreadIdByName(inferName_);
    preThreadId_ = GetAclLiteThreadIdByName(kPreName + to_string(channelId_));
    dataOutputThreadId_ =
        GetAclLiteThreadIdByName(kDataOutputName + to_string(channelId_));
    rtspDisplayThreadId_ =
        GetAclLiteThreadIdByName(kRtspDisplayName + to_string(channelId_));
    for (int i = 0; i < postThreadNum_; i++)
    {
        postThreadId_[i] = GetAclLiteThreadIdByName(
            kPostName + to_string(channelId_) + "_" + to_string(i));
        if (postThreadId_[i] == INVALID_INSTANCE_ID)
        {
            ACLLITE_LOG_ERROR(
                "%d postprocess instance id %d", i, postThreadId_[i]);
            return ACLLITE_ERROR;
        }
    }

    if ((selfThreadId_ == INVALID_INSTANCE_ID) ||
        (preThreadId_ == INVALID_INSTANCE_ID) ||
        (inferThreadId_ == INVALID_INSTANCE_ID) ||
        (dataOutputThreadId_ == INVALID_INSTANCE_ID))
    {
        ACLLITE_LOG_ERROR(
            "Self instance id %d, pre instance id %d, infer instance id %d,"
            "dataOutput instance id %d",
            selfThreadId_,
            preThreadId_,
            inferThreadId_,
            dataOutputThreadId_);
        return ACLLITE_ERROR;
    }
    int oneSecond = 1000;
    lastDecodeTime_ = 0;
    waitTime_ = oneSecond / framesPerSecond_;
    
    ACLLITE_LOG_INFO("DataInputThread initialized: frameSkip=%d (process 1 frame per %d frames)", 
                     frameSkip_, frameSkip_);

    return ACLLITE_OK;
}

AclLiteError DataInputThread::Process(int msgId, shared_ptr<void> msgData)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    shared_ptr<DetectDataMsg> detectDataMsg = make_shared<DetectDataMsg>();
    switch (msgId)
    {
    case MSG_APP_START:
        AppStart();
        break;
    case MSG_READ_FRAME:
        readFrameReady_ = true;
        if (readFrameReady_ && inferDoneReady_) {
            readFrameReady_ = false;
            inferDoneReady_ = false;
            MsgRead(detectDataMsg);
            MsgSend(detectDataMsg);
        }
        break;
    case MSG_INFER_DONE:
        inferDoneReady_ = true;
        if (readFrameReady_ && inferDoneReady_) {
            readFrameReady_ = false;
            inferDoneReady_ = false;
            MsgRead(detectDataMsg);
            MsgSend(detectDataMsg);
        }
        break;
    default:
        ACLLITE_LOG_ERROR("Detect Preprocess thread receive unknow msg %d",
                          msgId);
        break;
    }
    
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (frameCnt_ % 30 == 0)
        {
            ACLLITE_LOG_INFO("[DataInputThread] Process time: %ld ms", duration);
            // 每30帧打印一次所有线程的队列状态
            AclLiteApp &app = GetAclLiteAppInstance();
            app.PrintQueueStatus();
        }    return ACLLITE_OK;
}

AclLiteError DataInputThread::AppStart()
{
    AclLiteError ret = SendMessage(selfThreadId_, MSG_READ_FRAME, nullptr);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Process app start message failed, error %d", ret);
    }

    return ret;
}

AclLiteError DataInputThread::ReadPic(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    AclLiteError ret = ACLLITE_OK;
    if (frameCnt_ == fileVec_.size())
    {
        detectDataMsg->isLastFrame = true;
        return ACLLITE_OK;
    }
    ImageData jpgImg, dvppImg, decodedImg;
    string    picFile = fileVec_[frameCnt_];
    ret = ReadJpeg(jpgImg, picFile);
    if (ret == ACLLITE_ERROR)
    {
        ACLLITE_LOG_ERROR("Read Jpeg image failed");
        return ACLLITE_ERROR;
    }
    ret = CopyImageToDevice(dvppImg, jpgImg, runMode_, MEMORY_DVPP);
    if (ret == ACLLITE_ERROR)
    {
        ACLLITE_LOG_ERROR("Copy image to device failed");
        return ACLLITE_ERROR;
    }
    ret = dvpp_.JpegD(decodedImg, dvppImg);
    if (ret == ACLLITE_ERROR)
    {
        ACLLITE_LOG_ERROR("Pic decode failed");
        return ACLLITE_ERROR;
    }
    cv::Mat frame = cv::imread(picFile);
    detectDataMsg->decodedImg.push_back(decodedImg);
    detectDataMsg->frame.push_back(frame);
    return ACLLITE_OK;
}

AclLiteError
DataInputThread::ReadStream(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    // get time now
    timeval tv;
    gettimeofday(&tv, 0);
    int64_t now =
        ((int64_t)tv.tv_sec * kOneSec + (int64_t)tv.tv_usec) / kOneMSec;
    // get time last record
    if (lastDecodeTime_ == 0)
    {
        lastDecodeTime_ = now;
    }
    // calculate interval
    realWaitTime_ = (now - lastDecodeTime_);

    AclLiteError ret = ACLLITE_OK;
    ImageData    decodedImg;

    // 跳帧逻辑:跳过frameSkip_-1帧
    for (int i = 0; i < frameSkip_ - 1; i++)
    {
        ImageData skipFrame;
        ret = cap_->Read(skipFrame);
        if (ret == ACLLITE_ERROR_DECODE_FINISH)
        {
            detectDataMsg->isLastFrame = true;
            return ACLLITE_ERROR_DECODE_FINISH;
        }
        else if (ret != ACLLITE_OK)
        {
            detectDataMsg->isLastFrame = true;
            ACLLITE_LOG_ERROR("Read frame failed during skip, error %d", ret);
            return ACLLITE_ERROR;
        }
    }

    while (realWaitTime_ < waitTime_)
    {
        ret = cap_->Read(decodedImg);
        if (ret == ACLLITE_ERROR_DECODE_FINISH)
        {
            detectDataMsg->isLastFrame = true;
            return ACLLITE_ERROR_DECODE_FINISH;
        }
        else if (ret != ACLLITE_OK)
        {
            detectDataMsg->isLastFrame = true;
            ACLLITE_LOG_ERROR("Read frame failed, error %d", ret);
            return ACLLITE_ERROR;
        }
        // get time now
        gettimeofday(&tv, 0);
        now = ((int64_t)tv.tv_sec * kOneSec + (int64_t)tv.tv_usec) / kOneMSec;
        // calculate interval agian
        realWaitTime_ = (now - lastDecodeTime_);
    }

    ret = cap_->Read(decodedImg);
    if (ret == ACLLITE_ERROR_DECODE_FINISH)
    {
        detectDataMsg->isLastFrame = true;
        return ACLLITE_ERROR_DECODE_FINISH;
    }
    else if (ret != ACLLITE_OK)
    {
        detectDataMsg->isLastFrame = true;
        ACLLITE_LOG_ERROR("Read frame failed, error %d", ret);
        return ACLLITE_ERROR;
    }
    // get frame
    ImageData yuvImage;
    ret = CopyImageToLocal(yuvImage, decodedImg, runMode_);
    if (ret == ACLLITE_ERROR)
    {
        ACLLITE_LOG_ERROR("Copy image to host failed");
        return ACLLITE_ERROR;
    }
    cv::Mat frame;
    cv::Mat yuvimg(yuvImage.height * kYuvMultiplier / kYuvDivisor,
                   yuvImage.width,
                   CV_8UC1,
                   yuvImage.data.get());
    cv::cvtColor(yuvimg, frame, CV_YUV2BGR_NV12);
    detectDataMsg->decodedImg.push_back(decodedImg);
    detectDataMsg->frame.push_back(frame);
    lastDecodeTime_ = now;
    return ACLLITE_OK;
}

AclLiteError
DataInputThread::GetOneFrame(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    AclLiteError ret;
    if (inputDataType_ == "pic")
    {
        ret = ReadPic(detectDataMsg);
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("Read pic failed, error %d", ret);
            return ACLLITE_ERROR;
        }
    }
    else if (inputDataType_ == "video" || inputDataType_ == "rtsp")
    {
        ret = ReadStream(detectDataMsg);
        if (ret != ACLLITE_OK)
        {
            return ACLLITE_ERROR;
        }
    }
    else
    {
        ACLLITE_LOG_ERROR(
            "Invalid input data type, Please check your input file!");
        return ACLLITE_ERROR;
    }
    return ACLLITE_OK;
}

AclLiteError DataInputThread::MsgRead(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    // Record start timestamp for end-to-end latency measurement
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    detectDataMsg->startTimestamp = tv.tv_sec * 1000000 + tv.tv_usec;
    
    postproId_ = msgNum_ % postThreadNum_;
    detectDataMsg->isLastFrame = false;
    detectDataMsg->detectPreThreadId = preThreadId_;
    detectDataMsg->detectInferThreadId = inferThreadId_;
    detectDataMsg->detectPostThreadId = postThreadId_[postproId_];
    detectDataMsg->postId = postproId_;
    detectDataMsg->dataOutputThreadId = dataOutputThreadId_;
    detectDataMsg->rtspDisplayThreadId = rtspDisplayThreadId_;
    detectDataMsg->dataInputThreadId = selfThreadId_;
    detectDataMsg->deviceId = deviceId_;
    detectDataMsg->channelId = channelId_;
    detectDataMsg->msgNum = msgNum_;
    msgNum_++;
    GetOneFrame(detectDataMsg);
    if (detectDataMsg->isLastFrame)
    {
        return ACLLITE_OK;
    }
    frameCnt_++;
    while (frameCnt_ % batch_)
    {
        GetOneFrame(detectDataMsg);
        if (detectDataMsg->isLastFrame)
        {
            break;
        }
        frameCnt_++;
    }

    return ACLLITE_OK;
}

AclLiteError DataInputThread::MsgSend(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    AclLiteError ret;
    
    if (detectDataMsg->isLastFrame == false)
    {
        while (1)
        {
            ret = SendMessage(detectDataMsg->detectPreThreadId,
                              MSG_PREPROC_DETECTDATA,
                              detectDataMsg);
            if (ret == ACLLITE_ERROR_ENQUEUE)
            {
                usleep(kSleepTime);
                continue;
            }
            else if (ret == ACLLITE_OK)
            {
                break;
            }
            else
            {
                ACLLITE_LOG_ERROR("Send read frame message failed, error %d",
                                  ret);
                return ret;
            }
        }

        ret = SendMessage(selfThreadId_, MSG_READ_FRAME, nullptr);
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("Send read frame message failed, error %d", ret);
            return ret;
        }
    }
    else
    {
        for (int i = 0; i < postThreadNum_; i++)
        {
            while (1)
            {
                ret = SendMessage(detectDataMsg->detectPreThreadId,
                                  MSG_PREPROC_DETECTDATA,
                                  detectDataMsg);
                if (ret == ACLLITE_ERROR_ENQUEUE)
                {
                    usleep(kSleepTime);
                    continue;
                }
                else if (ret == ACLLITE_OK)
                {
                    break;
                }
                else
                {
                    ACLLITE_LOG_ERROR(
                        "Send read frame message failed, error %d", ret);
                    return ret;
                }
            }
        }
    }
    return ACLLITE_OK;
}
