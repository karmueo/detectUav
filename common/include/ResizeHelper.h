#pragma once
#include <cstdint>
#include "AclLiteUtils.h"
#include "acl/ops/acl_dvpp.h"

enum ResizeProcessType
{
    VPC_PT_DEFAULT = 0,
    VPC_PT_PADDING, // Resize with locked ratio and paste on upper left corner
    VPC_PT_FIT,     // Resize with locked ratio and paste on middle location
    VPC_PT_FILL,    // Resize with locked ratio and paste on whole locatin, the
                    // input image may be cropped
};

struct CropRoiConfig
{
    uint32_t left;
    uint32_t right;
    uint32_t down;
    uint32_t up;
};

class ResizeHelper
{
  public:
    /**
     * @brief Constructor
     * @param [in] stream: stream
     */
    ResizeHelper(aclrtStream        &stream,
                 acldvppChannelDesc *dvppChannelDesc,
                 uint32_t            width,
                 uint32_t            height);

    /**
     * @brief Destructor
     */
    ~ResizeHelper();

    /**
     * @brief dvpp global init
     * @return result
     */
    AclLiteError InitResource();

    /**
     * @brief init dvpp output para
     * @param [in] modelInputWidth: model input width
     * @param [in] modelInputHeight: model input height
     * @return result
     */
    AclLiteError InitOutputPara(int modelInputWidth, int modelInputHeight);

    /**
     * @brief gett dvpp output
     * @param [in] outputBuffer: pointer which points to dvpp output buffer
     * @param [out] outputSize: output size
     */
    void GetOutput(void **outputBuffer, int &outputSize);

    /**
     * @brief dvpp process
     * @return result
     */
    AclLiteError Process(ImageData        &resizedImage,
                         ImageData        &srcImage,
                         ResizeProcessType resizeType = VPC_PT_FIT);

  private:
    AclLiteError InitResizeResource(ImageData &inputImage);
    AclLiteError InitResizeInputDesc(ImageData &inputImage);
    AclLiteError InitResizeOutputDesc();

    /**
     * @brief 切片ROI
     *
     * @param input 输入，图片
     * @param processType 输入，缩放方式
     * @param cropRoi 输出，ROI
     */
    void GetCropRoi(const ImageData  &input,
                    ResizeProcessType processType,
                    CropRoiConfig    &cropRoi) const;

    /**
     * @brief 粘贴区域ROI
     *
     * @param input 输入，图片
     * @param processType 输入，缩放方式
     * @param pasteRoi 输出，ROI
     */
    void GetPasteRoi(const ImageData  &input,
                     ResizeProcessType processType,
                     CropRoiConfig    &pasteRoi) const;

    /**
     * @description: 保持宽高比的缩放，缩放后空白部分进行Padding
     * @param {CropRoiConfig} &cropRoi 输入，输入图片上的剪切ROI
     * @param {CropRoiConfig} &pasteRoi 输入，输出图片上的念贴ROI
     * @param {bool} withSynchronize 输入，是否同步等待
     * @return {*}
     * @author: 沈昌力
     */
    AclLiteError ResizeWithPadding(CropRoiConfig &cropRoi,
                                   CropRoiConfig &pasteRoi,
                                   bool           withSynchronize);

    void DestroyResizeResource();
    void DestroyResource();
    void DestroyOutputPara();

    aclrtStream          stream_;
    void                *vpcOutBufferDev_; // vpc output buffer
    acldvppPicDesc      *vpcInputDesc_;    // vpc input desc
    acldvppPicDesc      *vpcOutputDesc_;   // vpc output desc
    acldvppResizeConfig *resizeConfig_;
    acldvppChannelDesc  *dvppChannelDesc_;

    uint8_t                          *inDevBuffer_;      // input pic dev buffer
    uint32_t                          vpcOutBufferSize_; // vpc output size
    Resolution                        size_;
    std::shared_ptr<acldvppRoiConfig> cropAreaConfig_ = nullptr;
    std::shared_ptr<acldvppRoiConfig> pasteAreaConfig_ = nullptr;
};