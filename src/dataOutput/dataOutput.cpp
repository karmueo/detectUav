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
#include "dataOutput.h"
#include "AclLiteApp.h"
#include "drawing.h"
#include "label.h"
#include <chrono>
#include <cmath>
#include <sys/time.h>
#include <cstdio>

using namespace std;

namespace
{
const uint32_t kSleepTime = 500;
uint32_t       kWaitTime = 1000;
const uint32_t kOneSec = 1000000;
const uint32_t kOneMSec = 1000;
const uint32_t kCountFps = 100;
} // namespace

DataOutputThread::DataOutputThread(aclrtRunMode &runMode,
                                                                     string        outputDataType,
                                                                     string        outputPath,
                                                                     int           postThreadNum,
                                                                     VencConfig    vencConfig)
        : runMode_(runMode),
            outputDataType_(outputDataType),
            outputPath_(outputPath),
            shutdown_(0),
            postNum_(postThreadNum),
            g_vencConfig(vencConfig)
{
}

DataOutputThread::~DataOutputThread()
{
    if (outputDataType_ == "video")
    {
        outputVideo_.release();
    }
    dvpp_.DestroyResource();
}

AclLiteError DataOutputThread::SetOutputVideo()
{
    stringstream sstream;
    sstream.str("");
    sstream << outputPath_;
    int fps = g_vencConfig.outputFps > 0 ? g_vencConfig.outputFps : 15;
    outputVideo_.open(sstream.str(),
                      cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                      fps,
                      cv::Size(g_vencConfig.outputWidth, g_vencConfig.outputHeight));
    return ACLLITE_OK;
}

AclLiteError DataOutputThread::Init()
{
    // 初始化 dvpp 仅在 RTSP/HDMI 输出时需要（用于YUV Resize）
    if (outputDataType_ == "rtsp" || outputDataType_ == "hdmi") {
        AclLiteError dvppRet = dvpp_.Init("DVPP_CHNMODE_VPC");
        if (dvppRet != ACLLITE_OK) {
            ACLLITE_LOG_ERROR("DataOutput dvpp init failed, error %d", dvppRet);
            return ACLLITE_ERROR;
        }
    }
    if (outputDataType_ == "video")
    {
        AclLiteError ret = SetOutputVideo();
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("SetOutputVideo failed, error %d", ret);
            return ACLLITE_ERROR;
        }
    }
    if (outputDataType_ == "imshow")
    {
        kWaitTime = 1;
    }
    return ACLLITE_OK;
}

AclLiteError DataOutputThread::Process(int msgId, shared_ptr<void> data)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    AclLiteError ret = ACLLITE_OK;
    switch (msgId)
    {
    case MSG_OUTPUT_FRAME:
    {
        shared_ptr<DetectDataMsg> detectDataMsg =
            static_pointer_cast<DetectDataMsg>(data);
        if (detectDataMsg->decimatedFrame && detectDataMsg->reusePrevResult)
        {
            ApplyCachedResult(detectDataMsg);
            ProcessOutput(detectDataMsg);
        }
        else
        {
            RecordQueue(detectDataMsg);
            DataProcess();
        }
        break;
    }
    case MSG_ENCODE_FINISH:
        shutdown_++;
        if (shutdown_ == postNum_)
        {
            ShutDownProcess();
        }
        break;
    default:
        ACLLITE_LOG_INFO("Detect PostprocessThread thread ignore msg %d",
                         msgId);
        break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (msgId == MSG_OUTPUT_FRAME) {
        static int logCount = 0;
        if (++logCount % 30 == 0) {
            ACLLITE_LOG_INFO("[DataOutputThread] Process time: %ld ms", duration);
        }
    }

    return ret;
}

AclLiteError DataOutputThread::ShutDownProcess()
{
    for (int i = 0; i < postNum_; i++)
    {
        if (!postQueue_[i].empty())
        {
            shared_ptr<DetectDataMsg> detectDataMsg = postQueue_[i].front();
            ProcessOutput(detectDataMsg);
            postQueue_[i].pop();
            // processed one item
        }
    }
    if (outputDataType_ != "rtsp" && outputDataType_ != "hdmi")
    {
        SendMessage(g_MainThreadId, MSG_APP_EXIT, nullptr);
    }
    return ACLLITE_OK;
}

AclLiteError
DataOutputThread::RecordQueue(shared_ptr<DetectDataMsg> detectDataMsg)
{
    if (detectDataMsg->postId >= postNum_)
    {
        ACLLITE_LOG_ERROR("Support up to 4 post-processing of 1 inputdata.");
        return ACLLITE_ERROR;
    }
    postQueue_[detectDataMsg->postId].push(detectDataMsg);
    return ACLLITE_OK;
}

AclLiteError DataOutputThread::DataProcess()
{
    int flag = 0;
    for (int i = 0; i < postNum_; i++)
    {
        if (!postQueue_[i].empty())
            flag++;
    }
    if (flag == postNum_)
    {
        for (int i = 0; i < postNum_; i++)
        {
            shared_ptr<DetectDataMsg> detectDataMsg = postQueue_[i].front();
            ProcessOutput(detectDataMsg);
            postQueue_[i].pop();
        }
    }
    return ACLLITE_OK;
}

AclLiteError
DataOutputThread::ProcessOutput(shared_ptr<DetectDataMsg> detectDataMsg)
{
    // Calculate end-to-end latency
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    int64_t endTimestamp = tv.tv_sec * 1000000 + tv.tv_usec;
    int64_t latencyUs = endTimestamp - detectDataMsg->startTimestamp;
    double latencyMs = latencyUs / 1000.0;
    
    static int logCount = 0;
    if (++logCount % 30 == 0) {
        ACLLITE_LOG_INFO("[E2E Latency] Frame %d: %.2f ms (from DataInput to DataOutput)", 
                         detectDataMsg->msgNum, latencyMs);
    }
    
    // YUV color map for drawing (only draw on YUV, no BGR drawing)
    static const YUVColor kYUVColorTracking = YUVColor(149, 100, 237);  // Purple for tracking
    static const YUVColor kYUVColorDetection = YUVColor(215, 255, 0);   // Cyan for detections

    if (!detectDataMsg->decodedImg.empty())
    {
        // If tracking is active: only draw tracking box and text
        // Otherwise: draw all detection boxes and text
        if (detectDataMsg->trackingResult.isTracked)
        {
            const DetectionOBB &t = detectDataMsg->trackingResult.bbox;
            // If a detection in the current frame matches the tracked bbox,
            // prefer the detection's class and score; otherwise use the
            // initialized detection confidence and class stored in trackingResult.
            int chosen_class_id = t.class_id;
            // Attempt to find matching detection in the detections list
            for (size_t di = 0; di < detectDataMsg->detections.size(); ++di) {
                const auto &d = detectDataMsg->detections[di];
                float tracked_cx = (t.x0 + t.x1) * 0.5f;
                float tracked_cy = (t.y0 + t.y1) * 0.5f;
                if (d.x0 <= tracked_cx && tracked_cx <= d.x1 &&
                    d.y0 <= tracked_cy && tracked_cy <= d.y1) {
                    chosen_class_id = d.class_id;
                    break;
                }
            }

            const size_t labelCount = sizeof(::label) / sizeof(::label[0]);
            const std::string className =
                (chosen_class_id >= 0 && chosen_class_id < (int)labelCount) ?
                    ::label[chosen_class_id] : std::to_string(chosen_class_id);
            char labelText[128];
            // Only show class and current tracking confidence (curScore). Do not show initial detection confidence.
            snprintf(labelText, sizeof(labelText), "%s-%.2f",
                     className.c_str(),
                     detectDataMsg->trackingResult.curScore);
            
            // Draw on YUV only (no BGR drawing)
            DrawRect(detectDataMsg->decodedImg[0],
                     (int)t.x0, (int)t.y0,
                     (int)t.x1, (int)t.y1,
                     kYUVColorTracking, 2);
            DrawText(detectDataMsg->decodedImg[0],
                     (int)t.x0, std::max(0, (int)t.y0 - 30),
                     labelText, kYUVColorTracking, 24, 1.0f);
        }
        else if (!detectDataMsg->detections.empty())
        {
            // Draw all detections on YUV only (no BGR drawing)
            const size_t labelCount = sizeof(::label) / sizeof(::label[0]);
            for (size_t i = 0; i < detectDataMsg->detections.size(); ++i)
            {
                const auto &d = detectDataMsg->detections[i];
                if (detectDataMsg->filterStaticTargetEnabled &&
                    detectDataMsg->hasBlockedTarget)
                {
                    float det_w = d.x1 - d.x0;
                    float det_h = d.y1 - d.y0;
                    float det_cx = (d.x0 + d.x1) * 0.5f;
                    float det_cy = (d.y0 + d.y1) * 0.5f;
                    bool center_match =
                        std::fabs(det_cx - detectDataMsg->blockedCenterX) <=
                        detectDataMsg->staticCenterThreshold &&
                        std::fabs(det_cy - detectDataMsg->blockedCenterY) <=
                        detectDataMsg->staticCenterThreshold;
                    bool size_match =
                        std::fabs(det_w - detectDataMsg->blockedWidth) <=
                        detectDataMsg->staticSizeThreshold &&
                        std::fabs(det_h - detectDataMsg->blockedHeight) <=
                        detectDataMsg->staticSizeThreshold;
                    if (center_match && size_match)
                    {
                        continue;
                    }
                }
                const std::string detClassName =
                    (d.class_id >= 0 && d.class_id < (int)labelCount) ?
                        ::label[d.class_id] : std::to_string(d.class_id);
                char labelText[64];
                snprintf(labelText, sizeof(labelText), "%s-%.2f", detClassName.c_str(), d.score);
                
                DrawRect(detectDataMsg->decodedImg[0],
                         (int)d.x0, (int)d.y0,
                         (int)d.x1, (int)d.y1,
                         kYUVColorDetection, 2);
                DrawText(detectDataMsg->decodedImg[0],
                         (int)d.x0, std::max(0, (int)d.y0 - 30),
                         labelText, kYUVColorDetection, 24, 1.0f);
            }
        }
    }

    AclLiteError ret;
    if (outputDataType_ == "video")
    {
        ret = SaveResultVideo(detectDataMsg);
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("Draw classify result on video failed, error %d",
                              ret);
            return ACLLITE_ERROR;
        }
    }
    else if (outputDataType_ == "pic")
    {
        ret = SaveResultPic(detectDataMsg);
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("Draw classify result on pic failed, error %d",
                              ret);
            return ACLLITE_ERROR;
        }
    }
    else if (outputDataType_ == "stdout")
    {
        ret = PrintResult(detectDataMsg);
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("stdout result on screen failed, error %d", ret);
            return ACLLITE_ERROR;
        }
    }
    else if (outputDataType_ == "imshow")
    {
        ret = SendCVImshow(detectDataMsg);
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("Draw classify result on pic failed, error %d",
                              ret);
            return ACLLITE_ERROR;
        }
    }
    else if (outputDataType_ == "rtsp")
    {
        ret = SendImageToRtsp(detectDataMsg);
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("Send image to rtsp failed, error %d", ret);
            return ACLLITE_ERROR;
        }
    }
    else if (outputDataType_ == "hdmi")
    {
        ret = SendImageToHdmi(detectDataMsg);
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("Send image to hdmi failed, error %d", ret);
            return ACLLITE_ERROR;
        }
    }

    UpdateCachedResult(detectDataMsg);

    return ACLLITE_OK;
}

void DataOutputThread::UpdateCachedResult(
    const shared_ptr<DetectDataMsg> &detectDataMsg)
{
    CachedResult &cache = lastResults_[detectDataMsg->channelId];
    cache.detections = detectDataMsg->detections;
    cache.trackingResult = detectDataMsg->trackingResult;
    cache.textPrint = detectDataMsg->textPrint;
    cache.trackingActive = detectDataMsg->trackingActive;
    cache.trackingConfidence = detectDataMsg->trackingConfidence;
    cache.filterStaticTargetEnabled = detectDataMsg->filterStaticTargetEnabled;
    cache.hasBlockedTarget = detectDataMsg->hasBlockedTarget;
    cache.blockedCenterX = detectDataMsg->blockedCenterX;
    cache.blockedCenterY = detectDataMsg->blockedCenterY;
    cache.blockedWidth = detectDataMsg->blockedWidth;
    cache.blockedHeight = detectDataMsg->blockedHeight;
    cache.staticCenterThreshold = detectDataMsg->staticCenterThreshold;
    cache.staticSizeThreshold = detectDataMsg->staticSizeThreshold;
}

void DataOutputThread::ApplyCachedResult(
    shared_ptr<DetectDataMsg> &detectDataMsg)
{
    auto it = lastResults_.find(detectDataMsg->channelId);
    if (it == lastResults_.end())
    {
        return;
    }

    const CachedResult &cache = it->second;
    detectDataMsg->detections = cache.detections;
    detectDataMsg->trackingResult = cache.trackingResult;
    detectDataMsg->textPrint = cache.textPrint;
    detectDataMsg->trackingActive = cache.trackingActive;
    detectDataMsg->trackingConfidence = cache.trackingConfidence;
    detectDataMsg->filterStaticTargetEnabled = cache.filterStaticTargetEnabled;
    detectDataMsg->hasBlockedTarget = cache.hasBlockedTarget;
    detectDataMsg->blockedCenterX = cache.blockedCenterX;
    detectDataMsg->blockedCenterY = cache.blockedCenterY;
    detectDataMsg->blockedWidth = cache.blockedWidth;
    detectDataMsg->blockedHeight = cache.blockedHeight;
    detectDataMsg->staticCenterThreshold = cache.staticCenterThreshold;
    detectDataMsg->staticSizeThreshold = cache.staticSizeThreshold;
    detectDataMsg->hasTracking = cache.trackingResult.isTracked;
    detectDataMsg->trackScore = cache.trackingResult.curScore;
    detectDataMsg->trackInitScore = cache.trackingResult.initScore;
}

AclLiteError
DataOutputThread::SaveResultVideo(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    // Pre-allocate resized Mat to avoid per-frame allocation
    static cv::Mat resizedFrame;
    if (resizedFrame.empty() || resizedFrame.rows != (int)g_vencConfig.outputHeight || 
        resizedFrame.cols != (int)g_vencConfig.outputWidth) {
        resizedFrame = cv::Mat((int)g_vencConfig.outputHeight, (int)g_vencConfig.outputWidth, CV_8UC3);
    }
    
    for (int i = 0; i < detectDataMsg->frame.size(); i++)
    {
        cv::resize(detectDataMsg->frame[i], resizedFrame,
               cv::Size(g_vencConfig.outputWidth, g_vencConfig.outputHeight),
                   0, 0, cv::INTER_LINEAR);
        outputVideo_ << resizedFrame;
    }
    return ACLLITE_OK;
}

AclLiteError
DataOutputThread::SaveResultPic(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    // Use static char buffer to avoid repeated string allocations
    static char filepath[256];
    
    for (int i = 0; i < detectDataMsg->frame.size(); i++)
    {
        snprintf(filepath, sizeof(filepath), "../out/channel_%d_out_pic_%d%d.jpg",
                 detectDataMsg->channelId, detectDataMsg->msgNum, i);
        cv::imwrite(filepath, detectDataMsg->frame[i]);
    }
    return ACLLITE_OK;
}

AclLiteError
DataOutputThread::PrintResult(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    for (int i = 0; i < detectDataMsg->textPrint.size(); i++)
    {
        // get time now
        timeval tv;
        gettimeofday(&tv, 0);
        int64_t now =
            ((int64_t)tv.tv_sec * kOneSec + (int64_t)tv.tv_usec) / kOneMSec;
        if (lastDecodeTime_ == 0)
        {
            lastDecodeTime_ = now;
        }
        uint32_t lastIntervalTime = now - lastDecodeTime_;
        lastDecodeTime_ = now;
        detectDataMsg->textPrint[i] = detectDataMsg->textPrint[i] + "[" +
                                      to_string(lastIntervalTime) + "ms]";
        if (!(frameCnt_ % kCountFps))
        {
            if (lastRecordTime_ == 0)
            {
                lastRecordTime_ = now;
            }
            else
            {
                uint32_t fps = kCountFps / ((now - lastRecordTime_) / kOneMSec);
                lastRecordTime_ = now;
                detectDataMsg->textPrint[i] = detectDataMsg->textPrint[i] +
                                              "[fps:" + to_string(fps) + "]";
            }
        }
        frameCnt_++;
        cout << detectDataMsg->textPrint[i] << endl;
    }
    return ACLLITE_OK;
}

AclLiteError
DataOutputThread::SendCVImshow(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    // Pre-allocate resized Mat to avoid per-frame allocation
    static cv::Mat resizedFrame;
    if (resizedFrame.empty() || resizedFrame.rows != (int)g_vencConfig.outputHeight || 
        resizedFrame.cols != (int)g_vencConfig.outputWidth) {
        resizedFrame = cv::Mat((int)g_vencConfig.outputHeight, (int)g_vencConfig.outputWidth, CV_8UC3);
    }
    
    for (int i = 0; i < detectDataMsg->frame.size(); i++)
    {
        cv::resize(detectDataMsg->frame[i], resizedFrame,
               cv::Size(g_vencConfig.outputWidth, g_vencConfig.outputHeight));
        cv::imshow("frame", resizedFrame);
        cv::waitKey(kWaitTime);
    }
    return ACLLITE_OK;
}

AclLiteError
DataOutputThread::DisplayMsgSend(shared_ptr<DetectDataMsg> detectDataMsg)
{
    AclLiteError ret;
    int retryCount = 0;
    const int maxRetry = 3;  // 最多重试3次,避免阻塞
    
    while (retryCount < maxRetry)
    {
        if (outputDataType_ == "rtsp")
        {
            ret = SendMessage(detectDataMsg->rtspDisplayThreadId,
                              MSG_RTSP_DISPLAY,
                              detectDataMsg);
        }
        else if (outputDataType_ == "hdmi")
        {
            ret = SendMessage(detectDataMsg->hdmiDisplayThreadId,
                              MSG_HDMI_DISPLAY,
                              detectDataMsg);
        }
        if (ret == ACLLITE_ERROR_ENQUEUE)
        {
            retryCount++;
            if (retryCount >= maxRetry) {
                // 队列满,丢弃此帧
                static int dropCount = 0;
                if (++dropCount % 30 == 0) {
                    ACLLITE_LOG_INFO("[DataOutput] Dropped %d frames due to rtsp queue full", dropCount);
                }
                return ACLLITE_OK;  // 返回OK,继续处理下一帧
            }
            usleep(kSleepTime);
            continue;
        }
        else if (ret == ACLLITE_OK)
        {
            break;
        }
        else
        {
            ACLLITE_LOG_ERROR("Send rtsp display message failed, error %d",
                              ret);
            return ret;
        }
    }

    return ACLLITE_OK;
}

AclLiteError
DataOutputThread::SendImageToRtsp(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    // Resize 图像到推流分辨率: 使用 DVPP 进行 YUV Resize 并替换 decodedImg
    for (int i = 0; i < detectDataMsg->decodedImg.size(); i++) {
        ImageData &srcImg = detectDataMsg->decodedImg[i];
        if (srcImg.width != g_vencConfig.outputWidth || srcImg.height != g_vencConfig.outputHeight) {
            ImageData resizedImg;
            AclLiteError ret = dvpp_.Resize(resizedImg, srcImg, g_vencConfig.outputWidth, g_vencConfig.outputHeight);
            if (ret != ACLLITE_OK) {
                ACLLITE_LOG_ERROR("Dvpp resize in DataOutput failed, error %d", ret);
                return ACLLITE_ERROR;
            }
            // replace decoded image with resized one
            detectDataMsg->decodedImg[i] = resizedImg;
            // 同时更新对应的 cv::Mat frame，以便 video/show 使用（拷贝到Host）
            ImageData hostImg;
            ret = CopyImageToLocal(hostImg, resizedImg, runMode_);
            if (ret == ACLLITE_OK) {
                cv::Mat yuvimg(hostImg.height * 3 / 2,
                               hostImg.width,
                               CV_8UC1,
                               hostImg.data.get());
                cv::Mat bgr;
                cv::cvtColor(yuvimg, bgr, CV_YUV2BGR_NV12);
                detectDataMsg->frame[i] = bgr;
            }
        }
    }
    
    AclLiteError ret = DisplayMsgSend(detectDataMsg);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Send display msg failed");
        return ACLLITE_ERROR;
    }

    return ACLLITE_OK;
}

AclLiteError
DataOutputThread::SendImageToHdmi(shared_ptr<DetectDataMsg> &detectDataMsg)
{
    // 调整尺寸到HDMI输出分辨率（NV12）
    for (int i = 0; i < detectDataMsg->decodedImg.size(); i++) {
        ImageData &srcImg = detectDataMsg->decodedImg[i];
        if (srcImg.width != g_vencConfig.outputWidth || srcImg.height != g_vencConfig.outputHeight) {
            ImageData resizedImg;
            AclLiteError ret = dvpp_.Resize(resizedImg, srcImg, g_vencConfig.outputWidth, g_vencConfig.outputHeight);
            if (ret != ACLLITE_OK) {
                ACLLITE_LOG_ERROR("Dvpp resize in DataOutput (hdmi) failed, error %d", ret);
                return ACLLITE_ERROR;
            }
            detectDataMsg->decodedImg[i] = resizedImg;
        }
    }

    AclLiteError ret = DisplayMsgSend(detectDataMsg);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Send display msg to hdmi failed");
        return ACLLITE_ERROR;
    }

    return ACLLITE_OK;
}
