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
#ifndef PARAMS_H
#define PARAMS_H
#pragma once

#include "AclLiteImageProc.h"
#include "AclLiteModel.h"
#include "AclLiteType.h"
// Lightweight detection box for cross-thread messaging
struct DetectionOBB {
    float x0;
    float y0;
    float x1;
    float y1;
    float score;
    int   class_id;
};
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/opencv.hpp"
#include "X11/Xlib.h"
#include <iostream>
#include <memory>
#include <mutex>
#include <sys/timeb.h>
#include <unistd.h>
#include <vector>

namespace
{
const int MSG_APP_START = 1;
const int MSG_READ_FRAME = 2;
const int MSG_PREPROC_DETECTDATA = 3;
const int MSG_DO_DETECT_INFER = 4;
const int MSG_POSTPROC_DETECTDATA = 5;
const int MSG_OUTPUT_FRAME = 6;
const int MSG_ENCODE_FINISH = 7;
const int MSG_RTSP_DISPLAY = 8;
const int MSG_APP_EXIT = 9;
const int MSG_INFER_DONE = 10;
const int MSG_TRACK_DATA = 11; // detection -> tracker

const std::string kDataInputName = "dataInput";
const std::string kPreName = "pre";
const std::string kInferName = "infer";
const std::string kPostName = "post";
const std::string kDataOutputName = "dataOutput";
const std::string kRtspDisplayName = "rtspDisplay";
} // namespace

struct DetectDataMsg
{
    int      detectPreThreadId;
    int      detectInferThreadId;
    int      detectPostThreadId;
    int      dataOutputThreadId;
    int      rtspDisplayThreadId;
    int      trackThreadId = INVALID_INSTANCE_ID;      // tracking thread id (optional, -1 if unused)
    int      dataInputThreadId;  // data input thread id for flow control
    int      postId;
    uint32_t deviceId;
    uint32_t channelId; // record msg belongs to which rtsp/video channel
    bool isLastFrame;   // whether the last frame of rtsp/video of this channel
                        // has been decoded
    int msgNum;         // record frameID in rtsp/video of this channel
    int64_t startTimestamp;  // timestamp when frame processing starts (microseconds)
    std::vector<ImageData> decodedImg;    // original image (NV12)
    ImageData              modelInputImg; // image after detect preprocess
    std::vector<cv::Mat>   frame; // original image (BGR) needed by postprocess
    std::vector<InferenceOutput> inferenceOutput; // yolo detect output
    std::vector<std::string>     textPrint;
    // structured detections (per frame index), single-image pipelines use index 0
    std::vector<DetectionOBB>    detections;
    // tracking metadata
    bool   hasTracking = false;       // whether tracking is active
    float  trackInitScore = 0.0f;     // detection confidence at init
    float  trackScore = 0.0f;         // current tracking confidence
};

#endif