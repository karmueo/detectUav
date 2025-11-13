#include "ResizeHelper.h"
#include "AclLiteUtils.h"

using namespace std;

#define CONVERT_TO_ODD(NUM)                                                    \
    (((NUM) % 2 != 0) ? (NUM) : ((NUM) - 1)) // 将输入转换为奇数
#define CONVERT_TO_EVEN(NUM)                                                   \
    (((NUM) % 2 == 0) ? (NUM) : ((NUM) - 1)) // 将输入转换为偶数
#define DVPP_ALIGN_UP(x, align)                                                \
    ((((x) + ((align) - 1)) / (align)) * (align)) // 对齐到align

static auto g_roiConfigDeleter = [](acldvppRoiConfig *const p)
{ acldvppDestroyRoiConfig(p); };

ResizeHelper::ResizeHelper(aclrtStream        &stream,
                           acldvppChannelDesc *dvppChannelDesc,
                           uint32_t            width,
                           uint32_t            height)
    : stream_(stream),
      vpcOutBufferDev_(nullptr),
      vpcInputDesc_(nullptr),
      vpcOutputDesc_(nullptr),
      resizeConfig_(nullptr),
      dvppChannelDesc_(dvppChannelDesc),
      inDevBuffer_(nullptr),
      vpcOutBufferSize_(0)
{
    size_.width = width;
    size_.height = height;
}

ResizeHelper::~ResizeHelper() { DestroyResizeResource(); }

AclLiteError ResizeHelper::InitResizeInputDesc(ImageData &inputImage)
{
    uint32_t alignWidth = inputImage.alignWidth;
    uint32_t alignHeight = inputImage.alignHeight;
    if (alignWidth == 0 || alignHeight == 0)
    {
        ACLLITE_LOG_ERROR("Input image width %d or height %d invalid",
                          inputImage.width,
                          inputImage.height);
        return ACLLITE_ERROR_INVALID_ARGS;
    }

    uint32_t inputBufferSize = YUV420SP_SIZE(alignWidth, alignHeight);
    vpcInputDesc_ = acldvppCreatePicDesc();
    if (vpcInputDesc_ == nullptr)
    {
        ACLLITE_LOG_ERROR("Create dvpp pic desc failed");
        return ACLLITE_ERROR_CREATE_PIC_DESC;
    }

    acldvppSetPicDescData(vpcInputDesc_, inputImage.data.get());
    acldvppSetPicDescFormat(vpcInputDesc_, inputImage.format);
    acldvppSetPicDescWidth(vpcInputDesc_, inputImage.width);
    acldvppSetPicDescHeight(vpcInputDesc_, inputImage.height);
    acldvppSetPicDescWidthStride(vpcInputDesc_, alignWidth);
    acldvppSetPicDescHeightStride(vpcInputDesc_, alignHeight);
    acldvppSetPicDescSize(vpcInputDesc_, inputBufferSize);

    return ACLLITE_OK;
}

AclLiteError ResizeHelper::InitResizeOutputDesc()
{
    int resizeOutWidth = size_.width;
    int resizeOutHeight = size_.height;
    int resizeOutWidthStride = ALIGN_UP16(resizeOutWidth);
    int resizeOutHeightStride = ALIGN_UP2(resizeOutHeight);
    if (resizeOutWidthStride == 0 || resizeOutHeightStride == 0)
    {
        ACLLITE_LOG_ERROR("Align resize width(%d) and height(%d) failed",
                          size_.width,
                          size_.height);
        return ACLLITE_ERROR_INVALID_ARGS;
    }

    vpcOutBufferSize_ =
        YUV420SP_SIZE(resizeOutWidthStride, resizeOutHeightStride);
    aclError aclRet = acldvppMalloc(&vpcOutBufferDev_, vpcOutBufferSize_);
    if (aclRet != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Dvpp resize malloc output buffer failed, "
                          "size %d, error %d",
                          vpcOutBufferSize_,
                          aclRet);
        return ACLLITE_ERROR_MALLOC_DVPP;
    }

    vpcOutputDesc_ = acldvppCreatePicDesc();
    if (vpcOutputDesc_ == nullptr)
    {
        ACLLITE_LOG_ERROR("acldvppCreatePicDesc vpcOutputDesc_ failed");
        return ACLLITE_ERROR_CREATE_PIC_DESC;
    }

    acldvppSetPicDescData(vpcOutputDesc_, vpcOutBufferDev_);
    acldvppSetPicDescFormat(vpcOutputDesc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(vpcOutputDesc_, resizeOutWidth);
    acldvppSetPicDescHeight(vpcOutputDesc_, resizeOutHeight);
    acldvppSetPicDescWidthStride(vpcOutputDesc_, resizeOutWidthStride);
    acldvppSetPicDescHeightStride(vpcOutputDesc_, resizeOutHeightStride);
    acldvppSetPicDescSize(vpcOutputDesc_, vpcOutBufferSize_);

    return ACLLITE_OK;
}

AclLiteError ResizeHelper::InitResizeResource(ImageData &inputImage)
{
    resizeConfig_ = acldvppCreateResizeConfig();
    if (resizeConfig_ == nullptr)
    {
        ACLLITE_LOG_ERROR("Dvpp resize init failed for create config failed");
        return ACLLITE_ERROR_CREATE_RESIZE_CONFIG;
    }

    AclLiteError ret = InitResizeInputDesc(inputImage);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("InitResizeInputDesc failed");
        return ret;
    }

    ret = InitResizeOutputDesc();
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("InitResizeOutputDesc failed");
        return ret;
    }

    return ACLLITE_OK;
}

AclLiteError ResizeHelper::Process(ImageData        &resizedImage,
                                   ImageData        &srcImage,
                                   ResizeProcessType resizeType)
{
    AclLiteError atlRet = InitResizeResource(srcImage);
    if (atlRet != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Dvpp resize failed for init error");
        return atlRet;
    }

    // 等比缩放
    if (resizeType == VPC_PT_FIT)
    {
        // 等比缩放
        // 获取切片
        // When the processType is VPC_PT_FILL, the image will be cropped if the
        // image size is different from the target resolution
        CropRoiConfig cropRoi = {0};
        GetCropRoi(srcImage, resizeType, cropRoi);

        // 原图的宽高会按相同的比例调整大小
        // 裁剪后的图像根据processType粘贴在左上角或中间位置或整个位置
        CropRoiConfig pasteRoi = {0};
        GetPasteRoi(srcImage, resizeType, pasteRoi);

        atlRet = ResizeWithPadding(cropRoi, pasteRoi, true);
        if (atlRet != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("acldvppVpcResizeAsync failed, error: %d",
                              atlRet);
            return ACLLITE_ERROR_RESIZE_ASYNC;
        }
    }
    else
    {
        // 非等比缩放
        // resize pic
        aclError aclRet = acldvppVpcResizeAsync(dvppChannelDesc_,
                                                vpcInputDesc_,
                                                vpcOutputDesc_,
                                                resizeConfig_,
                                                stream_);
        if (aclRet != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("acldvppVpcResizeAsync failed, error: %d",
                              aclRet);
            return ACLLITE_ERROR_RESIZE_ASYNC;
        }

        aclRet = aclrtSynchronizeStream(stream_);
        if (aclRet != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("resize aclrtSynchronizeStream failed, error: %d",
                              aclRet);
            return ACLLITE_ERROR_SYNC_STREAM;
        }
    }

    resizedImage.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    resizedImage.width = size_.width;
    resizedImage.height = size_.height;
    resizedImage.alignWidth = ALIGN_UP16(size_.width);
    resizedImage.alignHeight = ALIGN_UP2(size_.height);
    resizedImage.size = vpcOutBufferSize_;
    resizedImage.data = SHARED_PTR_DVPP_BUF(vpcOutBufferDev_);

    DestroyResizeResource();

    return ACLLITE_OK;
}

void ResizeHelper::DestroyResizeResource()
{
    if (resizeConfig_ != nullptr)
    {
        (void)acldvppDestroyResizeConfig(resizeConfig_);
        resizeConfig_ = nullptr;
    }

    if (vpcInputDesc_ != nullptr)
    {
        (void)acldvppDestroyPicDesc(vpcInputDesc_);
        vpcInputDesc_ = nullptr;
    }

    if (vpcOutputDesc_ != nullptr)
    {
        (void)acldvppDestroyPicDesc(vpcOutputDesc_);
        vpcOutputDesc_ = nullptr;
    }
}

/**
 * @brief 切片ROI
 *
 * @param input 输入，图片
 * @param processType 输入，缩放方式
 * @param cropRoi 输出，ROI
 */
void ResizeHelper::GetCropRoi(const ImageData  &input,
                              ResizeProcessType processType,
                              CropRoiConfig    &cropRoi) const
{
    // When processType is not VPC_PT_FILL, crop area is the whole input image
    if (processType != VPC_PT_FILL)
    {
        cropRoi.right = CONVERT_TO_ODD(input.alignWidth - 1);
        cropRoi.down = CONVERT_TO_ODD(input.alignHeight - 1);
        return;
    }

    bool widthRatioSmaller = true;
    // The scaling ratio is based on the smaller ratio to ensure the smallest
    // edge to fill the targe edge
    float resizeRatio = static_cast<float>(input.alignWidth) / size_.width;
    if (resizeRatio > (static_cast<float>(input.alignHeight) / size_.height))
    {
        resizeRatio = static_cast<float>(input.alignHeight) / size_.height;
        widthRatioSmaller = false;
    }

    const int halfValue = 2;
    // 左上必须是偶数，右下必须是奇数，这是acl要求的
    if (widthRatioSmaller)
    {
        cropRoi.left = 0;
        cropRoi.right = CONVERT_TO_ODD(input.alignWidth - 1);
        cropRoi.up = CONVERT_TO_EVEN(static_cast<uint32_t>(
            (input.alignHeight - size_.height * resizeRatio) / halfValue));
        cropRoi.down = CONVERT_TO_ODD(input.alignHeight - cropRoi.up - 1);
        return;
    }

    cropRoi.up = 0;
    cropRoi.down = CONVERT_TO_ODD(input.height - 1);
    cropRoi.left = CONVERT_TO_EVEN(static_cast<uint32_t>(
        (input.alignWidth - size_.height * resizeRatio) / halfValue));
    cropRoi.right = CONVERT_TO_ODD(input.alignWidth - cropRoi.left - 1);
    return;
}

/**
 * @brief 粘贴区域ROI
 *
 * @param input 输入，图片
 * @param processType 输入，缩放方式
 * @param pasteRoi 输出，ROI
 */
void ResizeHelper::GetPasteRoi(const ImageData  &input,
                               ResizeProcessType processType,
                               CropRoiConfig    &pasteRoi) const
{
    if (processType == VPC_PT_FILL)
    {
        pasteRoi.right = CONVERT_TO_ODD(size_.width - 1);
        pasteRoi.down = CONVERT_TO_ODD(size_.height - 1);
        return;
    }

    bool widthRatioLarger = true;
    // 缩放比例以较大的比例为基础，以确保最大的边缘填充目标边缘
    float resizeRatio = static_cast<float>(input.width) / size_.width;
    if (resizeRatio < (static_cast<float>(input.height) / size_.height))
    {
        resizeRatio = static_cast<float>(input.height) / size_.height;
        widthRatioLarger = false;
    }

    // 左上角 roi 粘贴时 left and up 为 0
    if (processType == VPC_PT_PADDING)
    {
        pasteRoi.right = (input.width / resizeRatio) - 1;
        pasteRoi.down = (input.height / resizeRatio) - 1;
        pasteRoi.right = CONVERT_TO_ODD(pasteRoi.right);
        pasteRoi.down = CONVERT_TO_ODD(pasteRoi.down);
        return;
    }

    const int halfValue = 2;
    // 当 roi 粘贴在中间位置时，left and up 为 0
    if (widthRatioLarger)
    {
        pasteRoi.left = 0;
        pasteRoi.right = size_.width - 1;
        pasteRoi.up = (size_.height - (input.height / resizeRatio)) / halfValue;
        pasteRoi.down = size_.height - pasteRoi.up - 1;
    }
    else
    {
        pasteRoi.up = 0;
        pasteRoi.down = size_.height - 1;
        pasteRoi.left = (size_.width - (input.width / resizeRatio)) / halfValue;
        pasteRoi.right = size_.width - pasteRoi.left - 1;
    }

    // 左必须是偶数并对齐到 16，上必须是偶数，右和下必须是奇数，这是 acl 要求的
    pasteRoi.left = DVPP_ALIGN_UP(CONVERT_TO_EVEN(pasteRoi.left), 16);
    pasteRoi.right = CONVERT_TO_ODD(pasteRoi.right);
    pasteRoi.up = CONVERT_TO_EVEN(pasteRoi.up);
    pasteRoi.down = CONVERT_TO_ODD(pasteRoi.down);
    return;
}

/**
 * @description: 保持宽高比的缩放，缩放后空白部分进行Padding
 * @param {CropRoiConfig} &cropRoi 输入，输入图片上的剪切ROI
 * @param {CropRoiConfig} &pasteRoi 输入，输出图片上的念贴ROI
 * @param {bool} withSynchronize 输入，是否同步等待
 * @return {*}
 * @author: 沈昌力
 */
AclLiteError ResizeHelper::ResizeWithPadding(CropRoiConfig &cropRoi,
                                             CropRoiConfig &pasteRoi,
                                             bool           withSynchronize)
{
    acldvppRoiConfig *cropRoiCfg = acldvppCreateRoiConfig(
        cropRoi.left, cropRoi.right, cropRoi.up, cropRoi.down);
    if (cropRoiCfg == nullptr)
    {
        ACLLITE_LOG_ERROR("Failed to create dvpp roi config for corp area.");
        return ACLLITE_ERROR_CREATE_RESIZE_CONFIG;
    }
    cropAreaConfig_.reset(cropRoiCfg, g_roiConfigDeleter);

    acldvppRoiConfig *pastRoiCfg = acldvppCreateRoiConfig(
        pasteRoi.left, pasteRoi.right, pasteRoi.up, pasteRoi.down);
    if (pastRoiCfg == nullptr)
    {
        ACLLITE_LOG_ERROR("Failed to create dvpp roi config for paster area.");
        return ACLLITE_ERROR_CREATE_RESIZE_CONFIG;
    }
    pasteAreaConfig_.reset(pastRoiCfg, g_roiConfigDeleter);

    aclError ret = acldvppVpcCropAndPasteAsync(dvppChannelDesc_,
                                               vpcInputDesc_,
                                               vpcOutputDesc_,
                                               cropAreaConfig_.get(),
                                               pasteAreaConfig_.get(),
                                               stream_);
    if (ret != ACL_SUCCESS)
    {
        // release resource.
        ACLLITE_LOG_ERROR("Failed to crop and paste asynchronously, ret = %d.",
                          ret);
        return ACLLITE_ERROR_RESIZE_ASYNC;
    }
    if (withSynchronize)
    {
        ret = aclrtSynchronizeStream(stream_);
        if (ret != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("Failed tp synchronize stream, ret = %d.", ret);
            return ACLLITE_ERROR_RESIZE_ASYNC;
        }
    }
    return ACLLITE_OK;
}