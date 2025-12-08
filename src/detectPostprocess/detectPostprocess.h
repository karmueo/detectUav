#ifndef DETECTPOSTPROCESSTHREAD_H
#define DETECTPOSTPROCESSTHREAD_H
#pragma once

#include "AclLiteError.h"
#include "AclLiteImageProc.h"
#include "AclLiteThread.h"
#include "Params.h"
#include <unistd.h>

class DetectPostprocessThread : public AclLiteThread
{
  public:
    DetectPostprocessThread(uint32_t      modelWidth,
                            uint32_t      modelHeight,
                            aclrtRunMode &runMode,
                            uint32_t      batch,
                            int           targetClassId);
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
    aclrtRunMode runMode_;
    bool         sendLastBatch_;
    uint32_t     batch_;
    int          targetClassId_; // <0 means no class filtering
};

#endif
