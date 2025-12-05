#pragma once
#include "AclLiteThread.h"
#include "Params.h"
#include "pictortsp.h"
#include <sys/timeb.h>

class PushRtspThread : public AclLiteThread
{
  public:
    PushRtspThread(std::string rtspUrl, VencConfig vencConfig = VencConfig());
    ~PushRtspThread();
    AclLiteError Init();
    AclLiteError Process(int msgId, std::shared_ptr<void> msgData);
    AclLiteError
    DisplayMsgProcess(std::shared_ptr<DetectDataMsg> detectDataMsg);

  private:
    PicToRtsp   g_picToRtsp;
    uint64_t    g_frameSeq;
    std::string g_rtspUrl;
    VencConfig  g_vencConfig;
};
