#ifndef DETECTPOSTPROCESSTHREAD_H
#define DETECTPOSTPROCESSTHREAD_H
#pragma once

#include "AclLiteError.h"
#include "AclLiteImageProc.h"
#include "AclLiteThread.h"
#include "Params.h"
#include <unordered_set>
#include <vector>
#include <unistd.h>

class DetectPostprocessThread : public AclLiteThread
{
  public:
    DetectPostprocessThread(uint32_t      modelWidth,
                            uint32_t      modelHeight,
                            aclrtRunMode &runMode,
                            uint32_t      batch,
                            const std::vector<int> &targetClassIds,
                            ResizeProcessType resizeType,
                            bool          useNms);
    ~DetectPostprocessThread();

    AclLiteError Init();
    AclLiteError Process(int msgId, std::shared_ptr<void> data);

  private:
    AclLiteError
    InferOutputProcess(std::shared_ptr<DetectDataMsg> detectDataMsg);
    AclLiteError MsgSend(std::shared_ptr<DetectDataMsg> detectDataMsg);

  private:
    uint32_t     modelWidth_;
    uint32_t     modelHeight_;
    ResizeProcessType resizeType_; // 预处理缩放方式
    bool         useNms_;       // 是否使用NMS
    aclrtRunMode runMode_;
    bool         sendLastBatch_;
    uint32_t     batch_;
    std::vector<int>      targetClassIds_; // 过滤类别列表，空表示不过滤
    std::unordered_set<int> targetClassIdSet_; // 类别过滤集合，用于快速查找
    bool         targetClassChecked_ = false;
};

#endif
