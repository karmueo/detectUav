#ifndef DETECTPREPROCESSTHREAD_H
#define DETECTPREPROCESSTHREAD_H
#pragma once
#include "AclLiteImageProc.h"
#include "AclLiteThread.h"
#include "Params.h"
#include <unistd.h>

class DetectPreprocessThread : public AclLiteThread
{
  public:
    DetectPreprocessThread(uint32_t modelWidth,
                           uint32_t modelHeight,
                           uint32_t batch);
    ~DetectPreprocessThread();
    AclLiteError Init();
    AclLiteError Process(int msgId, std::shared_ptr<void> data);

  private:
    AclLiteError MsgProcess(std::shared_ptr<DetectDataMsg> detectDataMsg);
    AclLiteError MsgSend(std::shared_ptr<DetectDataMsg> detectDataMsg);

  private:
    uint32_t         modelWidth_;
    uint32_t         modelHeight_;
    AclLiteImageProc dvpp_;
    bool             isReleased;
    uint32_t         batch_;
};

#endif
