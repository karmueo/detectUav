/**
 * @file VencHelper.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cerrno>
#include <cstdint>
#include <cstring>

#include "VencHelper.h"

using namespace std;
namespace
{
uint32_t kKeyFrameInterval = 16;        // 关键帧间隔（GOP大小），影响：增大减少码率但增加寻址延迟，减小增加码率但改善实时性
uint32_t kRcMode = 2;                   // 码率控制模式（0:CBR恒定码率, 1:VBR可变码率, 2:AVBR自适应可变码率），影响：AVBR平衡质量和码率
uint32_t kMaxBitRate = 10000;           // 最大码率（kbps），影响：增大提高视频质量但增加带宽占用，减小降低质量但节省带宽
uint32_t kVencQueueSize = 256;          // 编码队列大小，影响：增大减少丢帧但增加内存占用和延迟，减小减少内存但可能丢帧
uint32_t kImageEnQueueRetryTimes = 3;   // 图像入队重试次数，影响：增大提高成功率但增加延迟，减小减少延迟但可能失败
uint32_t kEnqueueWait = 10000;          // 入队等待时间（微秒），影响：增大减少CPU占用但增加延迟，减小减少延迟但增加CPU占用
uint32_t kOutqueueWait = 10000;         // 出队等待时间（微秒），影响：增大减少CPU占用但增加延迟，减小减少延迟但增加CPU占用
uint32_t kAsyncWait = 10000;            // 异步等待时间（微秒），影响：增大减少CPU占用但增加初始化延迟，减小减少延迟但增加CPU占用
bool     g_runFlag = true;              // 全局运行标志，影响：控制编码器运行状态
} // namespace

VencHelper::VencHelper(VencConfig &vencInfo)
    : vencInfo_(vencInfo), status_(STATUS_VENC_INIT), vencProc_(nullptr),
      frameImageQueue_(kVencQueueSize)
{
}

VencHelper::~VencHelper() { DestroyResource(); }

AclLiteError VencHelper::Init()
{
    if (status_ != STATUS_VENC_INIT)
    {
        return ACLLITE_ERROR;
    }

    if (vencInfo_.context == nullptr)
    {
        aclError ret = aclrtGetCurrentContext(&(vencInfo_.context));
        if (ret != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("Get current context failed");
            return ACLLITE_ERROR_GET_ACL_CONTEXT;
        }
    }

    thread asyncVencTh = thread(VencHelper::AsyncVencThreadEntry, (void *)this);
    asyncVencTh.detach();
    while (status_ == STATUS_VENC_INIT)
    {
        usleep(kAsyncWait);
    }
    return (status_ == STATUS_VENC_WORK) ? ACLLITE_OK
                                         : ACLLITE_ERROR_VENC_STATUS;
}

/**
 * @brief 异步视频编码线程入口函数
 * 该函数在单独的线程中运行，负责初始化DVPP编码器并循环处理待编码的图像数据。
 * 如果初始化失败或处理过程中出错，会设置错误状态并退出线程。
 * @param arg 指向VencHelper对象的指针，用于访问编码配置和状态。
 */
void VencHelper::AsyncVencThreadEntry(void *arg)
{
    VencHelper *thisPtr = (VencHelper *)arg;
    DvppVenc    venc(thisPtr->vencInfo_);

    AclLiteError ret = venc.Init();
    if (ret != ACLLITE_OK)
    {
        thisPtr->SetStatus(STATUS_VENC_ERROR);
        ACLLITE_LOG_ERROR("Dvpp venc init acl resource failed, error %d", ret);
        return;
    }

    thisPtr->SetStatus(STATUS_VENC_WORK);
    while (thisPtr->GetStatus() == STATUS_VENC_WORK)
    {
        shared_ptr<ImageData> image = thisPtr->GetEncodeImage();
        if (image == nullptr)
        {
            usleep(kOutqueueWait);
            continue;
        }

        ret = venc.Process(*image.get());
        if (ret != ACLLITE_OK)
        {
            thisPtr->SetStatus(STATUS_VENC_ERROR);
            ACLLITE_LOG_ERROR("Dvpp venc image failed, error %d", ret);
            break;
        }
    }

    venc.Finish();
    thisPtr->SetStatus(STATUS_VENC_EXIT);
}

AclLiteError VencHelper::Process(ImageData &image)
{
    if (status_ != STATUS_VENC_WORK)
    {
        ACLLITE_LOG_ERROR("The venc(status %d) is not working", status_);
        return ACLLITE_ERROR_VENC_STATUS;
    }
    shared_ptr<ImageData> imagePtr = make_shared<ImageData>();
    imagePtr->format = image.format;
    imagePtr->width = image.width;
    imagePtr->height = image.height;
    imagePtr->alignWidth = image.alignWidth;
    imagePtr->alignHeight = image.alignHeight;
    imagePtr->size = image.size;
    imagePtr->data = image.data;

    for (uint32_t count = 0; count < kImageEnQueueRetryTimes; count++)
    {
        if (frameImageQueue_.Push(imagePtr))
        {
            return ACLLITE_OK;
        }
        usleep(kEnqueueWait);
    }
    ACLLITE_LOG_ERROR("Venc(%s) lost image for queue full",
                      vencInfo_.outFile.c_str());

    return ACLLITE_ERROR_VENC_QUEUE_FULL;
}

shared_ptr<ImageData> VencHelper::GetEncodeImage()
{
    shared_ptr<ImageData> image = frameImageQueue_.Pop();
    return image;
}

void VencHelper::DestroyResource() { vencProc_->Finish(); }

DvppVenc::DvppVenc(VencConfig &vencInfo)
    : vencInfo_(vencInfo), threadId_(0), vencChannelDesc_(nullptr),
      vencFrameConfig_(nullptr), inputPicDesc_(nullptr), vencStream_(nullptr),
      outFp_(nullptr), isFinished_(false)
{
}

DvppVenc::~DvppVenc() { DestroyResource(); }

/**
 * @brief NOTE: DVPP视频编码回调函数
 * 该函数由DVPP编码器异步调用，用于处理编码完成后的输出流数据。
 * 当编码成功时，将编码数据保存到文件或通过回调传递；编码失败时记录错误。
 * 同时负责释放输入图片描述符的内存资源。
 * @param input 指向输入图片描述符的指针，包含编码前的图像数据。
 * @param output 指向输出流描述符的指针，包含编码后的视频流数据。
 * @param userData 用户数据指针，指向DvppVenc对象，用于访问保存方法。
 */
void DvppVenc::Callback(acldvppPicDesc    *input,
                        acldvppStreamDesc *output,
                        void              *userData)
{
    // 从输出流描述符中获取编码后的数据指针
    void    *data = acldvppGetStreamDescData(output);
    // 获取编码结果码，0表示成功
    uint32_t retCode = acldvppGetStreamDescRetCode(output);
    if (retCode == 0)
    {
        // 编码成功，处理输出数据
        // 获取编码数据的大小
        uint32_t     size = acldvppGetStreamDescSize(output);
        // 将userData转换为DvppVenc对象指针
        DvppVenc    *venc = (DvppVenc *)userData;
        // 调用保存函数，将编码数据保存到文件或通过回调传递
        AclLiteError ret = venc->SaveVencFile(data, size);
        if (ret != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("Save venc file failed, error %d", ret);
        }
    }
    else
    {
        // 编码失败，记录错误码
        ACLLITE_LOG_ERROR("venc encode frame failed, ret = %u.", retCode);
    }
    // 获取输入图片描述符中的数据指针
    void *data_input = acldvppGetPicDescData(input);
    if (data_input != nullptr)
    {
        // 释放输入数据的设备内存
        acldvppFree(data_input);
    }
    // 销毁输入图片描述符
    acldvppDestroyPicDesc(input);
}

AclLiteError DvppVenc::SaveVencFile(void *vencData, uint32_t size)
{
    AclLiteError atlRet = ACLLITE_OK;
    void        *data = vencData;
    if (vencInfo_.runMode == ACL_HOST)
    {
        data = CopyDataToHost(vencData, size, vencInfo_.runMode, MEMORY_NORMAL);
    }

    // NOTE: 如果设置了回调函数，调用回调而不是写文件
    // 这里调用的是pictortsp.cpp中在AvInit中定义的g_vencConfig.dataCallback = VencDataCallbackStatic
    if (vencInfo_.dataCallback != nullptr)
    {
        vencInfo_.dataCallback(data, size, vencInfo_.callbackUserData);
        
        if (vencInfo_.runMode == ACL_HOST)
        {
            delete[] ((uint8_t *)data);
        }
        return ACLLITE_OK;
    }
    
    // 否则写入文件（原有逻辑）
    size_t ret = fwrite(data, 1, size, outFp_);
    if (ret != size)
    {
        ACLLITE_LOG_ERROR("Save venc file %s failed, need write %u bytes, "
                          "but only write %zu bytes, error: %s",
                          vencInfo_.outFile.c_str(),
                          size,
                          ret,
                          strerror(errno));
        atlRet = ACLLITE_ERROR_WRITE_FILE;
    }
    else
    {
        fflush(outFp_);
    }

    if (vencInfo_.runMode == ACL_HOST)
    {
        delete[] ((uint8_t *)data);
    }

    return atlRet;
}

AclLiteError DvppVenc::Init()
{
    // 只有在没有设置回调函数时才打开文件
    if (vencInfo_.dataCallback == nullptr)
    {
        outFp_ = fopen(vencInfo_.outFile.c_str(), "wb+");
        if (outFp_ == nullptr)
        {
            ACLLITE_LOG_ERROR("Open file %s failed, error %s",
                              vencInfo_.outFile.c_str(),
                              strerror(errno));
            return ACLLITE_ERROR_OPEN_FILE;
        }
    }
    else
    {
        outFp_ = nullptr;  // 使用回调时不需要文件
        ACLLITE_LOG_INFO("Using callback mode, no file output");
    }

    return InitResource();
}

void *DvppVenc::SubscribleThreadFunc(aclrtContext sharedContext)
{
    if (sharedContext == nullptr)
    {
        ACLLITE_LOG_ERROR("sharedContext can not be nullptr");
        return ((void *)(-1));
    }
    ACLLITE_LOG_INFO("use shared context for this thread");
    aclError ret = aclrtSetCurrentContext(sharedContext);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("aclrtSetCurrentContext failed, errorCode = %d",
                          static_cast<int32_t>(ret));
        return ((void *)(-1));
    }

    while (g_runFlag)
    {
        // Notice: timeout 1000ms
        (void)aclrtProcessReport(1000);
    }

    return (void *)0;
}

AclLiteError DvppVenc::InitResource()
{
    aclError aclRet = aclrtSetCurrentContext(vencInfo_.context);
    if (aclRet != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Set context for dvpp venc failed, error %d", aclRet);
        return ACLLITE_ERROR_SET_ACL_CONTEXT;
    }

    // create process callback thread
    AclLiteError ret = pthread_create(&threadId_,
                                      nullptr,
                                      &DvppVenc::SubscribleThreadFunc,
                                      vencInfo_.context);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Create venc subscrible thread failed, error %d",
                          ret);
        return ACLLITE_ERROR_CREATE_THREAD;
    }

    ret = CreateVencChannel();
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Create venc channel failed, error %d", ret);
        return ret;
    }

    aclRet = aclrtCreateStream(&vencStream_);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Create venc stream failed, error %d", aclRet);
        return ACLLITE_ERROR_CREATE_STREAM;
    }

    aclRet = aclrtSubscribeReport(threadId_, vencStream_);
    if (aclRet != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Venc subscrible report failed, error %d", aclRet);
        return ACLLITE_ERROR_SUBSCRIBE_REPORT;
    }

    ret = CreateFrameConfig();
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Create venc frame config failed, error %d", ret);
        return ret;
    }

    ACLLITE_LOG_INFO("venc init resource success");
    return ACLLITE_OK;
}

AclLiteError DvppVenc::CreateVencChannel()
{
    // create vdec channelDesc
    vencChannelDesc_ = aclvencCreateChannelDesc();
    if (vencChannelDesc_ == nullptr)
    {
        ACLLITE_LOG_ERROR("Create venc channel desc failed");
        return ACLLITE_ERROR_CREATE_VENC_CHAN_DESC;
    }

    aclvencSetChannelDescThreadId(vencChannelDesc_, threadId_);
    // NOTE: dvpp使用回调函数进行编码数据处理
    aclvencSetChannelDescCallback(vencChannelDesc_, &DvppVenc::Callback);
    aclvencSetChannelDescEnType(vencChannelDesc_, vencInfo_.enType);
    aclvencSetChannelDescPicFormat(vencChannelDesc_, vencInfo_.format);
    aclvencSetChannelDescPicWidth(vencChannelDesc_, vencInfo_.maxWidth);
    aclvencSetChannelDescPicHeight(vencChannelDesc_, vencInfo_.maxHeight);
    aclvencSetChannelDescKeyFrameInterval(vencChannelDesc_, kKeyFrameInterval);
    aclvencSetChannelDescRcMode(vencChannelDesc_, kRcMode);
    aclvencSetChannelDescMaxBitRate(vencChannelDesc_, kMaxBitRate);

    aclError ret = aclvencCreateChannel(vencChannelDesc_);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("fail to create venc channel");
        return ACLLITE_ERROR_CREATE_VENC_CHAN;
    }
    return ACLLITE_OK;
}

AclLiteError DvppVenc::CreateFrameConfig()
{
    vencFrameConfig_ = aclvencCreateFrameConfig();
    if (vencFrameConfig_ == nullptr)
    {
        ACLLITE_LOG_ERROR("Create frame config");
        return ACLLITE_ERROR_VENC_CREATE_FRAME_CONFIG;
    }

    AclLiteError ret = SetFrameConfig(0, 1);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Set frame config failed, error %d", ret);
        return ret;
    }

    return ACLLITE_OK;
}

AclLiteError DvppVenc::SetFrameConfig(uint8_t eos, uint8_t forceIFrame)
{
    // set eos
    aclError ret = aclvencSetFrameConfigEos(vencFrameConfig_, eos);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("fail to set eos, ret = %d", ret);
        return ACLLITE_ERROR_VENC_SET_EOS;
    }

    ret = aclvencSetFrameConfigForceIFrame(vencFrameConfig_, forceIFrame);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("fail to set venc ForceIFrame");
        return ACLLITE_ERROR_VENC_SET_IF_FRAME;
    }

    return ACLLITE_OK;
}

AclLiteError DvppVenc::Process(ImageData &image)
{
    // create picture desc
    AclLiteError ret = CreateInputPicDesc(image);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("fail to create picture description");
        return ret;
    }

    // send frame
    acldvppStreamDesc *outputStreamDesc = nullptr;

    ret = aclvencSendFrame(vencChannelDesc_,
                           inputPicDesc_,
                           static_cast<void *>(outputStreamDesc),
                           vencFrameConfig_,
                           (void *)this);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("send venc frame failed, error %d", ret);
        return ACLLITE_ERROR_VENC_SEND_FRAME;
    }

    return ACLLITE_OK;
}

AclLiteError DvppVenc::CreateInputPicDesc(ImageData &image)
{
    inputPicDesc_ = acldvppCreatePicDesc();
    if (inputPicDesc_ == nullptr)
    {
        ACLLITE_LOG_ERROR("Create input pic desc failed");
        return ACLLITE_ERROR_CREATE_PIC_DESC;
    }
    void    *inBufferDev_ = nullptr;
    uint32_t inBufferSize_ = image.size;
    auto     aclRet = acldvppMalloc(&inBufferDev_, inBufferSize_);
    if (vencInfo_.runMode != ACL_DEVICE)
    {
        aclRet = aclrtMemcpy(inBufferDev_,
                             inBufferSize_,
                             image.data.get(),
                             image.size,
                             ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR(
                "acl memcpy data to dev failed, image.size=%u, ret=%d.\n",
                image.size,
                aclRet);
            (void)acldvppFree(inBufferDev_);
            inBufferDev_ = nullptr;
            return false;
        }
    }
    else
    {
        // copy input to device memory
        aclRet = aclrtMemcpy(inBufferDev_,
                             inBufferSize_,
                             image.data.get(),
                             image.size,
                             ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (aclRet != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR(
                "acl memcpy data to dev failed, image.size=%u, ret=%d.\n",
                image.size,
                aclRet);
            (void)acldvppFree(inBufferDev_);
            inBufferDev_ = nullptr;
            return false;
        }
    }
    acldvppSetPicDescFormat(inputPicDesc_, vencInfo_.format);
    acldvppSetPicDescWidth(inputPicDesc_, image.width);
    acldvppSetPicDescHeight(inputPicDesc_, image.height);
    acldvppSetPicDescWidthStride(inputPicDesc_, ALIGN_UP16(image.width));
    acldvppSetPicDescHeightStride(inputPicDesc_, ALIGN_UP2(image.height));
    acldvppSetPicDescData(inputPicDesc_, inBufferDev_);
    acldvppSetPicDescSize(inputPicDesc_, image.size);

    return ACLLITE_OK;
}

void DvppVenc::Finish()
{
    if (isFinished_)
    {
        return;
    }

    // set frame config, eos frame
    AclLiteError ret = SetFrameConfig(1, 0);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Set eos frame config failed, error %d", ret);
        return;
    }

    // send eos frame
    ret = aclvencSendFrame(
        vencChannelDesc_, nullptr, nullptr, vencFrameConfig_, nullptr);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("fail to send eos frame, ret=%u", ret);
        return;
    }

    fclose(outFp_);
    outFp_ = nullptr;
    isFinished_ = true;
    ACLLITE_LOG_INFO("venc process success");

    return;
}

void DvppVenc::DestroyResource()
{
    Finish();

    if (vencChannelDesc_ != nullptr)
    {
        (void)aclvencDestroyChannel(vencChannelDesc_);
        (void)aclvencDestroyChannelDesc(vencChannelDesc_);
        vencChannelDesc_ = nullptr;
    }

    if (inputPicDesc_ != nullptr)
    {
        (void)acldvppDestroyPicDesc(inputPicDesc_);
        inputPicDesc_ = nullptr;
    }

    if (vencStream_ != nullptr)
    {
        aclError ret = aclrtDestroyStream(vencStream_);
        if (ret != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("Vdec destroy stream failed, error %d", ret);
        }
        vencStream_ = nullptr;
    }

    if (vencFrameConfig_ != nullptr)
    {
        (void)aclvencDestroyFrameConfig(vencFrameConfig_);
        vencFrameConfig_ = nullptr;
    }
}