/**
 * @file VencHelper.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef VENC_HELPER_H
#define VENC_HELPER_H
#pragma once
#include "AclLiteUtils.h"
#include "ThreadSafeQueue.h"
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include <cstdint>
#include <iostream>
#include <thread>

class DvppVenc
{
  public:
    DvppVenc(VencConfig &vencConfig);
    ~DvppVenc();

    AclLiteError Init();
    AclLiteError Process(ImageData &image);
    void         Finish();
    void         StopSubscribeThread() { runFlag_ = false; }

  private:
    AclLiteError InitResource();
    AclLiteError CreateVencChannel();
    AclLiteError CreateInputPicDesc(ImageData &image);
    AclLiteError CreateFrameConfig();
    AclLiteError SetFrameConfig(uint8_t eos, uint8_t forceIFrame);
    AclLiteError SaveVencFile(void *vencData, uint32_t size);
    void         DestroyResource();

    static void
    Callback(acldvppPicDesc *input, acldvppStreamDesc *output, void *userData);
    static void *SubscribleThreadFunc(void *arg);

  private:
    VencConfig vencInfo_;

    pthread_t           threadId_;
    aclvencChannelDesc *vencChannelDesc_;
    aclvencFrameConfig *vencFrameConfig_;
    acldvppPicDesc     *inputPicDesc_;
    aclrtStream         vencStream_;

    FILE *outFp_;
    bool  isFinished_;
    bool  runFlag_;  // 实例级运行标志
};

class VencHelper
{
  public:
    VencHelper(VencConfig &vencConfig);
    ~VencHelper();

    AclLiteError Init();
    AclLiteError Process(ImageData &image);

    void       SetStatus(VencStatus status) { status_ = status; }
    void       DestroyResource();
    VencStatus GetStatus() { return status_; }
    // 获取待编码输入队列大小（frameImageQueue_）
    uint32_t   GetFrameQueueSize() { return frameImageQueue_.Size(); }

  private:
    static void                AsyncVencThreadEntry(void *arg);
    std::shared_ptr<ImageData> GetEncodeImage();

  private:
    VencConfig                                  vencInfo_;
    VencStatus                                  status_;
    DvppVenc                                   *vencProc_;
    ThreadSafeQueue<std::shared_ptr<ImageData>> frameImageQueue_;  // 待编码输入队列
};

#endif