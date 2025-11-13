#include "detectPreprocess.h"
#include "AclLiteApp.h"
#include <chrono>
#include "Params.h"
#include <sys/timeb.h>

using namespace std;

namespace
{
const uint32_t kSleepTime = 500;
}

DetectPreprocessThread::DetectPreprocessThread(uint32_t modelWidth,
                                               uint32_t modelHeight,
                                               uint32_t batch)
    : modelWidth_(modelWidth),
      modelHeight_(modelHeight),
      isReleased(false),
      batch_(batch)
{
}

DetectPreprocessThread::~DetectPreprocessThread()
{
    if (!isReleased)
    {
        dvpp_.DestroyResource();
    }
    isReleased = true;
}

AclLiteError DetectPreprocessThread::Init()
{
    AclLiteError aclRet = dvpp_.Init("DVPP_CHNMODE_VPC");
    if (aclRet)
    {
        ACLLITE_LOG_ERROR("Dvpp init failed, error %d", aclRet);
        return ACLLITE_ERROR;
    }

    return ACLLITE_OK;
}

AclLiteError DetectPreprocessThread::Process(int msgId, shared_ptr<void> data)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    switch (msgId)
    {
    case MSG_PREPROC_DETECTDATA:
        MsgProcess(static_pointer_cast<DetectDataMsg>(data));
        MsgSend(static_pointer_cast<DetectDataMsg>(data));
        break;
    default:
        ACLLITE_LOG_INFO("Detect Preprocess thread ignore msg %d", msgId);
        break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (msgId == MSG_PREPROC_DETECTDATA) {
        static int logCount = 0;
        if (++logCount % 30 == 0) {
            ACLLITE_LOG_INFO("[DetectPreprocessThread] Process time: %ld ms", duration);
        }
    }

    return ACLLITE_OK;
}

AclLiteError
DetectPreprocessThread::MsgProcess(shared_ptr<DetectDataMsg> detectDataMsg)
{
    AclLiteError ret;

    uint32_t modelInputSize = YUV420SP_SIZE(modelWidth_, modelHeight_) * batch_;
    void    *buf = nullptr;
    ret = aclrtMalloc(&buf, modelInputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if ((buf == nullptr) || (ret != ACL_ERROR_NONE))
    {
        ACLLITE_LOG_ERROR("Malloc classify inference input buffer failed, "
                          "error %d",
                          ret);
        return ACLLITE_ERROR;
    }
    uint8_t *batchBuffer = (uint8_t *)buf;
    int32_t  setValue = 0;
    aclrtMemset(batchBuffer, modelInputSize, setValue, modelInputSize);

    size_t pos = 0;
    for (int i = 0; i < detectDataMsg->decodedImg.size(); i++)
    {
        ImageData resizedImg;
        ret = dvpp_.Resize(resizedImg,
                           detectDataMsg->decodedImg[i],
                           modelWidth_,
                           modelHeight_);
        if (ret == ACLLITE_ERROR)
        {
            ACLLITE_LOG_ERROR("Resize image failed");
            return ACLLITE_ERROR;
        }
        uint32_t dataSize = YUV420SP_SIZE(modelWidth_, modelHeight_);
        ret = aclrtMemcpy(batchBuffer + pos,
                          dataSize,
                          resizedImg.data.get(),
                          resizedImg.size,
                          ACL_MEMCPY_DEVICE_TO_DEVICE);
        pos = pos + dataSize;
    }

    detectDataMsg->modelInputImg.data = SHARED_PTR_DEV_BUF(batchBuffer);
    detectDataMsg->modelInputImg.size = modelInputSize;
    return ACLLITE_OK;
}

AclLiteError
DetectPreprocessThread::MsgSend(shared_ptr<DetectDataMsg> detectDataMsg)
{
    while (1)
    {
        AclLiteError ret = SendMessage(detectDataMsg->detectInferThreadId,
                                       MSG_DO_DETECT_INFER,
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
            ACLLITE_LOG_ERROR("Send read frame message failed, error %d", ret);
            return ret;
        }
    }
    return ACLLITE_OK;
}
