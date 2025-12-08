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

* File main.cpp
* Description: dvpp sample main func
*/

#include "AclLiteApp.h"
#include "AclLiteResource.h"
#include "AclLiteThread.h"
#include "AclLiteUtils.h"
#include "Params.h"
#include "dataInput/dataInput.h"
#include "dataOutput/dataOutput.h"
#include "detectInference/detectInference.h"
#include "detectPostprocess/detectPostprocess.h"
#include "detectPreprocess/detectPreprocess.h"
#include "tracking/tracking.h"
#include "pushrtsp/pushrtspthread.h"
#include "hdmiOutput/hdmiOutputThread.h"
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <json/json.h>

using namespace std;
namespace
{
uint32_t             kExitCount = 0;
string               kJsonFile = "";
vector<aclrtContext> kContext;
uint32_t             kBatch = 1;
int                  kPostNum = 1;
int                  kFramesPerSecond = 1000;
uint32_t             kMsgQueueSize = 3;
uint32_t             argNum = 2;
} // namespace

int MainThreadProcess(uint32_t msgId, shared_ptr<void> msgData, void *userData)
{
    if (msgId == MSG_APP_EXIT)
    {
        kExitCount--;
    }
    if (!kExitCount)
    {
        AclLiteApp &app = GetAclLiteAppInstance();
        app.WaitEnd();
        ACLLITE_LOG_INFO("Receive exit message, exit now");
    }

    return ACLLITE_OK;
}

void CreateALLThreadInstance(vector<AclLiteThreadParam> &threadTbl,
                             AclLiteResource            &aclDev)
{
    aclrtRunMode runMode = aclDev.GetRunMode();
    Json::Reader reader;
    Json::Value  root;
    ifstream     srcFile(kJsonFile, ios::binary);
    if (!srcFile.is_open())
    {
        ACLLITE_LOG_ERROR("Fail to open test.json");
        return;
    }
    if (reader.parse(srcFile, root))
    {
        for (int i = 0; i < root["device_config"].size(); i++)
        {
            // Create context on the device
            uint32_t deviceId = root["device_config"][i]["device_id"].asInt();
            if (deviceId < 0)
            {
                ACLLITE_LOG_ERROR("Invaild deviceId: %d", deviceId);
                return;
            }
            aclrtContext context = aclDev.GetContextByDevice(deviceId);
            if (context == nullptr)
            {
                ACLLITE_LOG_ERROR("Get acl context in device %d failed", i);
                return;
            }
            kContext.push_back(context);
            for (int j = 0; j < root["device_config"][i]["model_config"].size();
                 j++)
            {
                // Get modelwidth, modelheight, modelpath, and thread name for
                // each model thread
                // Generate infer thread name using Params.h constant and indexes
                // to ensure uniqueness across devices and models
                string inferName = kInferName + to_string(deviceId) + "_" + to_string(j);
                string modelPath =
                    root["device_config"][i]["model_config"][j]["model_path"]
                        .asString();
                uint32_t modelWidth =
                    root["device_config"][i]["model_config"][j]["model_width"]
                        .asInt();
                uint32_t modelHeigth =
                    root["device_config"][i]["model_config"][j]["model_heigth"]
                        .asInt();
                if (root["device_config"][i]["model_config"][j]["model_batch"]
                        .type() != Json::nullValue)
                {
                    kBatch = root["device_config"][i]["model_config"][j]
                                 ["model_batch"]
                                     .asInt();
                }
                if (root["device_config"][i]["model_config"][j]["postnum"]
                        .type() != Json::nullValue)
                {
                    kPostNum =
                        root["device_config"][i]["model_config"][j]["postnum"]
                            .asInt();
                }

                if (root["device_config"][i]["model_config"][j]
                        ["frames_per_second"]
                            .type() != Json::nullValue)
                {
                    kFramesPerSecond = root["device_config"][i]["model_config"]
                                           [j]["frames_per_second"]
                                               .asInt();
                }

                // New: frame_decimation means how many frames to skip after
                // processing one frame (0 => no skip). Default is 0, and each
                // io_info 可以单独覆盖。
                int modelFrameDecimation = 0;
                if (root["device_config"][i]["model_config"][j]["frame_decimation"]
                        .type() != Json::nullValue)
                {
                    modelFrameDecimation = root["device_config"][i]["model_config"][j]
                                                   ["frame_decimation"]
                                                       .asInt();
                    if (modelFrameDecimation < 0)
                    {
                        ACLLITE_LOG_WARNING("frame_decimation is negative, clamping to 0");
                        modelFrameDecimation = 0;
                    }
                }
                // Note: legacy field 'frame_skip' is no longer supported. Use 'frame_decimation'.

                if (modelWidth < 0 || modelHeigth < 0 || kBatch < 1 ||
                    kPostNum < 1 || kFramesPerSecond < 1)
                {
                    ACLLITE_LOG_ERROR(
                        "Invaild model config is given! modelWidth: %d, "
                        "modelHeigth: %d,"
                        "batch: %d, postNum: %d, framesPerSecond: %d",
                        modelWidth,
                        modelHeigth,
                        kBatch,
                        kPostNum,
                        kFramesPerSecond);
                    return;
                }
                // Create inferThread
                AclLiteThreadParam inferParam;
                inferParam.threadInst = new DetectInferenceThread(modelPath);
                inferParam.threadInstName.assign(inferName.c_str());
                inferParam.context = context;
                inferParam.runMode = runMode;
                threadTbl.push_back(inferParam);
                // Read track configuration from model_config -> track_config (same level as io_info)
                bool enableTrackingModel = true; // default behavior remains true
                string trackModelPathModel = "";
                Json::Value trackingConfigModel;
                bool hasTrackConfigModel = false;
                if (root["device_config"][i]["model_config"][j]["track_config"].type() != Json::nullValue)
                {
                    Json::Value trackConf = root["device_config"][i]["model_config"][j]["track_config"];
                    hasTrackConfigModel = true;
                    if (trackConf["enable_tracking"].type() != Json::nullValue)
                    {
                        enableTrackingModel = trackConf["enable_tracking"].asBool();
                    }
                    if (trackConf["track_model_path"].type() != Json::nullValue)
                    {
                        trackModelPathModel = trackConf["track_model_path"].asString();
                    }
                    if (trackConf["tracking_config"].type() != Json::nullValue)
                    {
                        trackingConfigModel = trackConf["tracking_config"];
                    }
                }

                for (int k = 0;
                     k < root["device_config"][i]["model_config"][j]["io_info"]
                             .size();
                     k++)
                {
                    // Get all information for each input data:
                    string inputPath = root["device_config"][i]["model_config"]
                                           [j]["io_info"][k]["input_path"]
                                               .asString();
                    string inputType = root["device_config"][i]["model_config"]
                                           [j]["io_info"][k]["input_type"]
                                               .asString();
                    string outputPath = root["device_config"][i]["model_config"]
                                            [j]["io_info"][k]["output_path"]
                                                .asString();
                    string outputType = root["device_config"][i]["model_config"]
                                            [j]["io_info"][k]["output_type"]
                                                .asString();
                    uint32_t channelId =
                        root["device_config"][i]["model_config"][j]["io_info"]
                            [k]["channel_id"]
                                .asInt();
                    
                    // 解析 RTSP 和 H264 编码配置
                    VencConfig vencConfig;
                    vencConfig.maxWidth = modelWidth;
                    vencConfig.maxHeight = modelHeigth;
                    
                    // 解析 rtsp_config（仅 rtsp 输出生效）
                    if (outputType == "rtsp" &&
                        root["device_config"][i]["model_config"][j]["io_info"][k]["rtsp_config"].type() != Json::nullValue)
                    {
                        Json::Value rtspCfg = root["device_config"][i]["model_config"][j]["io_info"][k]["rtsp_config"];
                        if (rtspCfg["output_width"].type() != Json::nullValue)
                        {
                            vencConfig.outputWidth = rtspCfg["output_width"].asUInt();
                        }
                        if (rtspCfg["output_height"].type() != Json::nullValue)
                        {
                            vencConfig.outputHeight = rtspCfg["output_height"].asUInt();
                        }
                        if (rtspCfg["output_fps"].type() != Json::nullValue)
                        {
                            vencConfig.outputFps = rtspCfg["output_fps"].asUInt();
                            if (vencConfig.outputFps < 1 || vencConfig.outputFps > 60)
                            {
                                ACLLITE_LOG_WARNING("Output FPS %u out of range [1,60], using default 25", vencConfig.outputFps);
                                vencConfig.outputFps = 25;
                            }
                        }
                        if (rtspCfg["transport"].type() != Json::nullValue)
                        {
                            vencConfig.rtspTransport = rtspCfg["transport"].asString();
                        }
                        if (rtspCfg["buffer_size"].type() != Json::nullValue)
                        {
                            vencConfig.rtspBufferSize = rtspCfg["buffer_size"].asUInt();
                        }
                        if (rtspCfg["max_delay"].type() != Json::nullValue)
                        {
                            vencConfig.rtspMaxDelay = rtspCfg["max_delay"].asUInt();
                        }
                    }

                    // 解析 hdmi_config（仅 hdmi 输出生效）
                    if (outputType == "hdmi" &&
                        root["device_config"][i]["model_config"][j]["io_info"][k]["hdmi_config"].type() != Json::nullValue)
                    {
                        Json::Value hdmiCfg = root["device_config"][i]["model_config"][j]["io_info"][k]["hdmi_config"];
                        if (hdmiCfg["output_width"].type() != Json::nullValue)
                        {
                            vencConfig.outputWidth = hdmiCfg["output_width"].asUInt();
                        }
                        if (hdmiCfg["output_height"].type() != Json::nullValue)
                        {
                            vencConfig.outputHeight = hdmiCfg["output_height"].asUInt();
                        }
                        if (hdmiCfg["output_fps"].type() != Json::nullValue)
                        {
                            vencConfig.outputFps = hdmiCfg["output_fps"].asUInt();
                            if (vencConfig.outputFps < 1 || vencConfig.outputFps > 60)
                            {
                                ACLLITE_LOG_WARNING("HDMI output FPS %u out of range [1,60], using default 25", vencConfig.outputFps);
                                vencConfig.outputFps = 25;
                            }
                        }
                    }
                    
                    // 解析 h264_config
                    if (root["device_config"][i]["model_config"][j]["io_info"][k]["h264_config"].type() != Json::nullValue)
                    {
                        Json::Value h264Cfg = root["device_config"][i]["model_config"][j]["io_info"][k]["h264_config"];
                        if (h264Cfg["gop_size"].type() != Json::nullValue)
                        {
                            vencConfig.gopSize = h264Cfg["gop_size"].asUInt();
                            if (vencConfig.gopSize < 1 || vencConfig.gopSize > 300)
                            {
                                ACLLITE_LOG_WARNING("GOP size %u out of range [1,300], using default 16", vencConfig.gopSize);
                                vencConfig.gopSize = 16;
                            }
                        }
                        if (h264Cfg["rc_mode"].type() != Json::nullValue)
                        {
                            vencConfig.rcMode = h264Cfg["rc_mode"].asUInt();
                            if (vencConfig.rcMode > 2)
                            {
                                ACLLITE_LOG_WARNING("RC mode %u invalid (0=CBR,1=VBR,2=AVBR), using default 2", vencConfig.rcMode);
                                vencConfig.rcMode = 2;
                            }
                        }
                        if (h264Cfg["max_bitrate"].type() != Json::nullValue)
                        {
                            vencConfig.maxBitrate = h264Cfg["max_bitrate"].asUInt();
                            if (vencConfig.maxBitrate < 500 || vencConfig.maxBitrate > 50000)
                            {
                                ACLLITE_LOG_WARNING("Bitrate %u kbps out of range [500,50000], using default 10000", vencConfig.maxBitrate);
                                vencConfig.maxBitrate = 10000;
                            }
                        }
                        if (h264Cfg["profile"].type() != Json::nullValue)
                        {
                            std::string profile = h264Cfg["profile"].asString();
                            if (profile == "baseline")
                            {
                                vencConfig.enType = H264_BASELINE_LEVEL;
                            }
                            else if (profile == "main")
                            {
                                vencConfig.enType = H264_MAIN_LEVEL;
                            }
                            else if (profile == "high")
                            {
                                vencConfig.enType = H264_HIGH_LEVEL;
                            }
                        }
                    }
                    string dataInputName =
                        kDataInputName + to_string(channelId);
                    string preName = kPreName + to_string(channelId);
                    string dataOutputName =
                        kDataOutputName + to_string(channelId);
                    string rtspDisplayName =
                        kRtspDisplayName + to_string(channelId);

                    // per-channel frame decimation override (defaults to model-level)
                    int channelFrameDecimation = modelFrameDecimation;
                    if (root["device_config"][i]["model_config"][j]["io_info"][k]["frame_decimation"]
                            .type() != Json::nullValue)
                    {
                        channelFrameDecimation =
                            root["device_config"][i]["model_config"][j]["io_info"][k]
                                ["frame_decimation"]
                                    .asInt();
                        if (channelFrameDecimation < 0)
                        {
                            ACLLITE_LOG_WARNING(
                                "io_info[%d] frame_decimation is negative, clamping to 0",
                                channelId);
                            channelFrameDecimation = 0;
                        }
                    }

                    // Create Thread for the input data:
                    AclLiteThreadParam dataInputParam;
                    dataInputParam.threadInst =
                        new DataInputThread(deviceId,
                                            channelId,
                                            runMode,
                                            inputType,
                                            inputPath,
                                            inferName,
                                            kPostNum,
                                            kBatch,
                                            kFramesPerSecond,
                                            channelFrameDecimation,
                                            outputType);
                    dataInputParam.threadInstName.assign(dataInputName.c_str());
                    dataInputParam.context = context;
                    dataInputParam.runMode = runMode;
                    dataInputParam.queueSize = kMsgQueueSize;
                    threadTbl.push_back(dataInputParam);

                    AclLiteThreadParam detectPreParam;
                    detectPreParam.threadInst = new DetectPreprocessThread(
                        modelWidth, modelHeigth, kBatch);
                    detectPreParam.threadInstName.assign(preName.c_str());
                    detectPreParam.context = context;
                    detectPreParam.runMode = runMode;
                    detectPreParam.queueSize = kMsgQueueSize;
                    threadTbl.push_back(detectPreParam);
                    for (int m = 0; m < kPostNum; m++)
                    {
                        string postName = kPostName + to_string(channelId) +
                                          "_" + to_string(m);
                        AclLiteThreadParam detectPostParam;
                        detectPostParam.threadInst =
                            new DetectPostprocessThread(
                                modelWidth, modelHeigth, runMode, kBatch);
                        detectPostParam.threadInstName.assign(postName.c_str());
                        detectPostParam.context = context;
                        detectPostParam.runMode = runMode;
                        threadTbl.push_back(detectPostParam);
                    }

                    // 创建单目标跟踪线程（Tracking），每个通道一个实例
                    // Use model-level tracking configuration
                    bool enableTracking = enableTrackingModel;
                    string trackModelPath = trackModelPathModel;
                    
                    // If model-level config absent, fallback to io_info-level
                    if (!hasTrackConfigModel)
                    {
                        if (root["device_config"][i]["model_config"][j]["io_info"][k]["enable_tracking"].type() != Json::nullValue)
                        {
                            enableTracking = root["device_config"][i]["model_config"][j]["io_info"][k]["enable_tracking"].asBool();
                        }
                        if (enableTracking && root["device_config"][i]["model_config"][j]["io_info"][k]["track_model_path"].type() != Json::nullValue)
                        {
                            trackModelPath = root["device_config"][i]["model_config"][j]["io_info"][k]["track_model_path"].asString();
                        }
                    }
                    
                    string trackName = kTrackName + to_string(channelId);
                    Tracking* trackingInst = nullptr;
                    if (enableTracking)
                    {
                        trackingInst = new Tracking(trackModelPath); // 使用配置文件中的模型路径
                        
                        // 读取并设置跟踪配置参数
                        Json::Value trackingConfig;
                        if (trackingConfigModel.type() != Json::nullValue)
                        {
                            trackingConfig = trackingConfigModel;
                        }
                        else if (root["device_config"][i]["model_config"][j]["io_info"][k]["tracking_config"].type() != Json::nullValue)
                        {
                            trackingConfig = root["device_config"][i]["model_config"][j]["io_info"][k]["tracking_config"];
                        }
                        if (trackingConfig.type() != Json::nullValue)
                        {
                            
                            if (trackingConfig["confidence_active_threshold"].type() != Json::nullValue)
                            {
                                float threshold = trackingConfig["confidence_active_threshold"].asFloat();
                                trackingInst->setConfidenceActiveThreshold(threshold);
                                ACLLITE_LOG_INFO("Set tracking confidence_active_threshold=%.2f for channel %d", threshold, channelId);
                            }
                            
                            if (trackingConfig["confidence_redetect_threshold"].type() != Json::nullValue)
                            {
                                float threshold = trackingConfig["confidence_redetect_threshold"].asFloat();
                                trackingInst->setConfidenceRedetectThreshold(threshold);
                                ACLLITE_LOG_INFO("Set tracking confidence_redetect_threshold=%.2f for channel %d", threshold, channelId);
                            }
                            
                            if (trackingConfig["max_track_loss_frames"].type() != Json::nullValue)
                            {
                                int maxLossFrames = trackingConfig["max_track_loss_frames"].asInt();
                                trackingInst->setMaxTrackLossFrames(maxLossFrames);
                                ACLLITE_LOG_INFO("Set tracking max_track_loss_frames=%d for channel %d", maxLossFrames, channelId);
                            }
                            
                            if (trackingConfig["score_decay_factor"].type() != Json::nullValue)
                            {
                                float decay = trackingConfig["score_decay_factor"].asFloat();
                                trackingInst->setMaxScoreDecay(decay);
                                ACLLITE_LOG_INFO("Set tracking score_decay_factor=%.2f for channel %d", decay, channelId);
                            }
                        }
                        
                        AclLiteThreadParam trackParam;
                        trackParam.threadInst = trackingInst;
                        trackParam.threadInstName.assign(trackName.c_str());
                        trackParam.context = context;
                        trackParam.runMode = runMode;
                        trackParam.queueSize = kMsgQueueSize;
                        threadTbl.push_back(trackParam);
                    }

                    AclLiteThreadParam dataOutputParam;
                    dataOutputParam.threadInst = new DataOutputThread(
                        runMode, outputType, outputPath, kPostNum, vencConfig);
                    dataOutputParam.threadInstName.assign(
                        dataOutputName.c_str());
                    dataOutputParam.context = context;
                    dataOutputParam.runMode = runMode;
                    threadTbl.push_back(dataOutputParam);

                    if (outputType == "rtsp")
                    {
                        AclLiteThreadParam rtspDisplayThreadParam;
                        rtspDisplayThreadParam.threadInst = new PushRtspThread(
                            outputPath + to_string(channelId), vencConfig);
                        rtspDisplayThreadParam.threadInstName.assign(
                            rtspDisplayName.c_str());
                        rtspDisplayThreadParam.context = context;
                        rtspDisplayThreadParam.runMode = runMode;
                        rtspDisplayThreadParam.queueSize = 1000;  // 增大队列避免积压
                        threadTbl.push_back(rtspDisplayThreadParam);
                    }
                    else if (outputType == "hdmi")
                    {
                        AclLiteThreadParam hdmiDisplayParam;
                        hdmiDisplayParam.threadInst = new HdmiOutputThread(runMode, vencConfig);
                        hdmiDisplayParam.threadInstName.assign(
                            (kHdmiDisplayName + to_string(channelId)).c_str());
                        hdmiDisplayParam.context = context;
                        hdmiDisplayParam.runMode = runMode;
                        hdmiDisplayParam.queueSize = 1000; // 与 RTSP 输出一致，避免 decimation 模式下排队失败
                        threadTbl.push_back(hdmiDisplayParam);
                    }
                    kExitCount++;
                }
            }
        }
    }
    srcFile.close();
}

void ExitApp(AclLiteApp &app, vector<AclLiteThreadParam> &threadTbl)
{
    for (int i = 0; i < threadTbl.size(); i++)
    {
        aclrtSetCurrentContext(threadTbl[i].context);
        delete threadTbl[i].threadInst;
    }
    app.Exit();

    for (int i = 0; i < kContext.size(); i++)
    {
        aclrtDestroyContext(kContext[i]);
        if (i != 0)
        {
            aclrtResetDevice(i);
        }
    }
}

void StartApp(AclLiteResource &aclDev)
{
    vector<AclLiteThreadParam> threadTbl;
    CreateALLThreadInstance(threadTbl, aclDev);
    AclLiteApp  &app = CreateAclLiteAppInstance();
    AclLiteError ret = app.Start(threadTbl);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Start app failed, error %d", ret);
        ExitApp(app, threadTbl);
        return;
    }

    for (int i = 0; i < threadTbl.size(); i++)
    {
        ret = SendMessage(threadTbl[i].threadInstId, MSG_APP_START, nullptr);
    }
    app.Wait(MainThreadProcess, nullptr);
    ExitApp(app, threadTbl);
    return;
}

int main(int argc, char *argv[])
{
    // Check input args: the path of configure json
    if ((argc != static_cast<int>(argNum)) || (argv[1] == nullptr))
    {
        ACLLITE_LOG_ERROR("Please input: ./main <json_dir>");
        return ACLLITE_ERROR;
    }
    kJsonFile = string(argv[1]);
    AclLiteResource aclDev = AclLiteResource();
    AclLiteError    ret = aclDev.Init();
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Init app failed");
    }
    StartApp(aclDev);
    return ACLLITE_OK;
}
