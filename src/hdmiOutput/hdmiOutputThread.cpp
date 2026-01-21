/**
 * HDMI output thread: initializes VO/HDMI and sends NV12 frames to display.
 */
#include "hdmiOutputThread.h"
#include "AclLiteApp.h"
#include "AclLiteUtils.h"
#include <chrono>
#include <cstring>
#include <unistd.h>

namespace
{
// GetHdmiIntfSyncBySize 根据期望输出尺寸选择 HDMI 时序。
// Args:
//   width: 期望输出宽度。
//   height: 期望输出高度。
// Returns:
//   HDMI 时序枚举值，默认返回 1080P60。
hi_vo_intf_sync GetHdmiIntfSyncBySize(uint32_t width, uint32_t height)
{
    if (width == 1280 && height == 720)
    {
        return HI_VO_OUT_720P60;
    }
    if (width == 1920 && height == 1080)
    {
        return HI_VO_OUT_1080P60;
    }
    ACLLITE_LOG_WARNING("Unsupported HDMI size %ux%u, fallback to 1080P60",
                        width,
                        height);
    return HI_VO_OUT_1080P60;
}

// vo_mipi_get_sync_info 根据时序填充同步信息。
// Args:
//   intf_sync: HDMI 时序枚举。
//   sync_info: 输出同步信息。
void vo_mipi_get_sync_info(hi_vo_intf_sync intf_sync,
                           vo_mst_sync_info *sync_info)
{
    sync_info->intf_sync = intf_sync;
    if (intf_sync == HI_VO_OUT_720P60)
    {
        sync_info->name = const_cast<char *>("720P@60");
        sync_info->width = 1280;
        sync_info->height = 720;
        sync_info->frame_rate = 60;
        return;
    }
    sync_info->name = const_cast<char *>("1080P@60");
    sync_info->width = 1920;
    sync_info->height = 1080;
    sync_info->frame_rate = 60;
}

hi_void vo_init_dev(hi_s32 dev, hi_vo_intf_type intf_type, hi_vo_intf_sync intf_sync)
{
    hi_vo_pub_attr pub_attr;
    pub_attr.bg_color = 0xffffff;
    pub_attr.intf_sync = intf_sync;
    pub_attr.intf_type = intf_type;
    VO_CHECK_RET(hi_mpi_vo_set_pub_attr(dev, &pub_attr), "hi_mpi_vo_set_pub_attr");
    VO_CHECK_RET(hi_mpi_vo_enable(dev), "hi_mpi_vo_enable");
}

hi_void vo_init_layer(hi_s32 layer, hi_u32 img_height, hi_u32 img_width, hi_u32 frame_rate)
{
    hi_vo_video_layer_attr layer_attr;
    layer_attr.double_frame_en = HI_FALSE;
    layer_attr.cluster_mode_en = HI_FALSE;
    layer_attr.dst_dynamic_range = HI_DYNAMIC_RANGE_SDR8;
    layer_attr.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    layer_attr.display_buf_len = 4;
    layer_attr.partition_mode = HI_VO_PARTITION_MODE_SINGLE;
    layer_attr.compress_mode = HI_COMPRESS_MODE_NONE;
    layer_attr.display_rect.width = img_width;
    layer_attr.display_rect.height = img_height;
    layer_attr.display_rect.x = 0;
    layer_attr.display_rect.y = 0;
    layer_attr.img_size.width = img_width;
    layer_attr.img_size.height = img_height;
    layer_attr.display_frame_rate = frame_rate;
    VO_CHECK_RET(hi_mpi_vo_set_video_layer_attr(layer, &layer_attr), "hi_mpi_vo_set_video_layer_attr");
    VO_CHECK_RET(hi_mpi_vo_enable_video_layer(layer), "hi_mpi_vo_enable_video_layer");
}

hi_void vo_init_chn(hi_s32 layer, hi_u32 img_height, hi_u32 img_width)
{
    hi_vo_chn_attr chn_attr;
    chn_attr.rect.x = 0;
    chn_attr.rect.y = 0;
    chn_attr.rect.width = VO_ALIGN_BACK(img_width, VO_MST_ALIGN_2);
    chn_attr.rect.height = VO_ALIGN_BACK(img_height, VO_MST_ALIGN_2);
    chn_attr.priority = 0;
    chn_attr.deflicker_en = HI_FALSE;

    VO_CHECK_RET(hi_mpi_vo_set_chn_attr(layer, 0, &chn_attr), "hi_mpi_vo_set_chn_attr");
    VO_CHECK_RET(hi_mpi_vo_enable_chn(layer, 0), "hi_mpi_vo_enable_chn");
}

void user_set_avi_infoframe_pattern(hi_hdmi_avi_infoframe *avi, int timing)
{
    avi->timing_mode = static_cast<hi_hdmi_video_format>(timing);
    avi->color_space = HI_HDMI_COLOR_SPACE_RGB444;
    avi->active_info_present = HI_FALSE; // Active Format Aspect Ratio
    avi->bar_info = HI_HDMI_BAR_INFO_NOT_VALID;
    avi->scan_info = HI_HDMI_SCAN_INFO_NO_DATA;
    avi->colorimetry = HI_HDMI_COMMON_COLORIMETRY_ITU601;
    avi->ex_colorimetry = HI_HDMI_COMMON_COLORIMETRY_XVYCC_601;
    avi->aspect_ratio = HI_HDMI_PIC_ASPECT_RATIO_4TO3;
    avi->active_aspect_ratio = HI_HDMI_ACTIVE_ASPECT_RATIO_SAME_PIC;
    avi->pic_scaling = HI_HDMI_PIC_NON_UNIFORM_SCALING;
    avi->rgb_quant = HI_HDMI_RGB_QUANT_FULL_RANGE;
    avi->is_it_content = HI_FALSE;
    avi->pixel_repetition = HI_HDMI_PIXEL_REPET_NO;
    avi->content_type = HI_HDMI_CONTNET_PHOTO;
    avi->ycc_quant = HI_HDMI_YCC_QUANT_FULL_RANGE;
    avi->line_n_end_of_top_bar = 0;
    avi->line_n_start_of_bot_bar = 0;
    avi->pixel_n_end_of_left_bar = 0;
    avi->pixel_n_start_of_right_bar = 0;
}

void user_set_audio_infoframe(hi_hdmi_audio_infoframe *audio)
{
    audio->chn_cnt = HI_HDMI_AUDIO_CHN_CNT_2;
    audio->coding_type = HI_HDMI_AUDIO_CODING_PCM;
    audio->sample_size = HI_HDMI_AUDIO_SAMPLE_SIZE_16;
    audio->sampling_freq = HI_HDMI_AUDIO_SAMPLE_FREQ_48000;
    audio->chn_alloc = 0; /* Channel/Speaker Allocation.Range [0,255] */
    audio->level_shift = HI_HDMI_LEVEL_SHIFT_VAL_0_DB;
    audio->lfe_playback_level = HI_HDMI_LFE_PLAYBACK_NO;
    audio->down_mix_inhibit = HI_FALSE;
}

hi_s32 hi_mpi_hdmi_set_info(hi_hdmi_attr attr, int hdmi_timing)
{
    hi_hdmi_infoframe infoframe;
    hi_s32 ret = hi_mpi_hdmi_set_attr(HI_HDMI_ID_0, &attr);
    if (ret != HI_SUCCESS) {
        ACLLITE_LOG_ERROR("hi_mpi_hdmi_set_attr error: %d", ret);
        return HI_FAILURE;
    }

    infoframe.infoframe_type = HI_INFOFRAME_TYPE_AVI;
    hi_hdmi_avi_infoframe *avi = &infoframe.infoframe_unit.avi_infoframe;
    user_set_avi_infoframe_pattern(avi, hdmi_timing);
    avi->color_space = HI_HDMI_COLOR_SPACE_YCBCR444;
    ret = hi_mpi_hdmi_set_infoframe(HI_HDMI_ID_0, &infoframe);
    if (ret != HI_SUCCESS) {
        ACLLITE_LOG_ERROR("[avi]hi_mpi_hdmi_set_infoframe error: %d", ret);
        return HI_FAILURE;
    }

    infoframe.infoframe_type = HI_INFOFRAME_TYPE_AUDIO;
    hi_hdmi_audio_infoframe *audio = &infoframe.infoframe_unit.audio_infoframe;
    user_set_audio_infoframe(audio);
    ret = hi_mpi_hdmi_set_infoframe(HI_HDMI_ID_0, &infoframe);
    if (ret != HI_SUCCESS) {
        ACLLITE_LOG_ERROR("[audio]hi_mpi_hdmi_set_infoframe error: %d", ret);
        return HI_FAILURE;
    }
    sleep(1);
    return HI_SUCCESS;
}

hi_void hi_mpi_hdmi_init_sample(void)
{
    hi_s32 ret = hi_mpi_hdmi_init();
    if (ret != HI_SUCCESS) {
        ACLLITE_LOG_ERROR("hi_mpi_hdmi_init_sample error: %d", ret);
        return;
    }
    ret = hi_mpi_hdmi_open(HI_HDMI_ID_0);
    if (ret != HI_SUCCESS) {
        ACLLITE_LOG_ERROR("hi_mpi_hdmi_open error: %d", ret);
        return;
    }
}

hi_s32 hi_mpi_hdmi_avi_infoframe_colorspace(int hdmi_timing, int pix_clk)
{
    hi_hdmi_attr attr;
    attr.hdmi_en = HI_TRUE;
    attr.video_format = HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE;
    attr.deep_color_mode = HI_HDMI_DEEP_COLOR_24BIT;
    attr.audio_en = HI_TRUE;
    attr.sample_rate = HI_HDMI_SAMPLE_RATE_48K;
    attr.bit_depth = HI_HDMI_BIT_DEPTH_16;
    attr.auth_mode_en = HI_FALSE;
    attr.deep_color_adapt_en = HI_TRUE; // 根据vo输出自动调整hdmi色域打开
    attr.pix_clk = pix_clk;

    hi_s32 ret = hi_mpi_hdmi_set_info(attr, hdmi_timing);
    if (ret != HI_SUCCESS) {
        return HI_FAILURE;
    }

    ret = hi_mpi_hdmi_set_info(attr, hdmi_timing);
    if (ret != HI_SUCCESS) {
        return HI_FAILURE;
    }

    ret = hi_mpi_hdmi_start(HI_HDMI_ID_0);
    if (ret != HI_SUCCESS) {
        ACLLITE_LOG_ERROR("hi_mpi_hdmi_start error: %d", ret);
        return HI_FAILURE;
    }
    sleep(1);

    return HI_SUCCESS;
}

hi_u64 vo_mst_get_vb_blk_size(hi_u32 width, hi_u32 height)
{
    hi_u64 vb_blk_size;
    hi_u32 align_width;
    hi_u32 align_height;
    hi_u32 head_size;

    align_width = VO_TEST_ALIGN_BACK(width, VO_MST_ALIGN_16);
    align_height = VO_TEST_ALIGN_BACK(height, VO_MST_ALIGN_2);
    head_size = VO_MST_ALIGN_16 * align_height; // compress header stride 16
    vb_blk_size = (align_width * align_height + head_size) * 2; // NV12 => Y+UV
    return vb_blk_size;
}

hi_void vo_mipi_sys_init(hi_u32 img_height, hi_u32 img_width)
{
    (void)img_height;
    (void)img_width;
    hi_mpi_sys_exit();
    VO_CHECK_RET(hi_mpi_sys_init(), "sys init");
}

hi_void vo_mipi_sys_exit(hi_void)
{
    VO_CHECK_RET(hi_mpi_sys_exit(), "sys exit");
}

hi_s32 vo_mipi_create_vb_pool(hi_u32 img_height, hi_u32 img_width, hi_u32 *blk_handle)
{
    hi_vb_pool_config pool_cfg;
    (hi_void)memset(&pool_cfg, 0, sizeof(hi_vb_pool_config));

    pool_cfg.blk_size = vo_mst_get_vb_blk_size(img_width, img_height);
    pool_cfg.blk_cnt = 10;
    pool_cfg.remap_mode = VB_REMAP_MODE_NONE;

    hi_u32 pool_val = hi_mpi_vo_create_pool(pool_cfg.blk_size);
    *blk_handle = pool_val;
    if (pool_val == static_cast<hi_u32>(-1)) {
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_void vo_init_user_frame(hi_u32 vb_pool_val, hi_u32 img_height, hi_u32 img_width, hi_video_frame_info *user_frame)
{
    hi_u32 luma_size = 0;
    hi_u32 chroma_size = 0;

    user_frame->v_frame.field = HI_VIDEO_FIELD_FRAME;
    user_frame->v_frame.compress_mode = HI_COMPRESS_MODE_NONE;
    user_frame->v_frame.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    user_frame->v_frame.video_format = HI_VIDEO_FORMAT_LINEAR;
    user_frame->v_frame.color_gamut = HI_COLOR_GAMUT_BT709;
    user_frame->v_frame.dynamic_range = HI_DYNAMIC_RANGE_SDR8;
    user_frame->v_frame.height = img_height;
    user_frame->v_frame.width = img_width;
    user_frame->v_frame.width_stride[0] = VO_TEST_ALIGN_BACK(img_width, VO_MST_ALIGN_2);
    user_frame->v_frame.width_stride[1] = VO_TEST_ALIGN_BACK(img_width, VO_MST_ALIGN_2);
    user_frame->v_frame.time_ref = 0;
    user_frame->v_frame.pts = 0;

    luma_size = user_frame->v_frame.width * user_frame->v_frame.height;
    luma_size = VO_TEST_ALIGN_BACK(user_frame->v_frame.width, 2) * user_frame->v_frame.height;
    chroma_size = luma_size / 2;

    user_frame->pool_id = (vb_pool_val >> 16);
    user_frame->v_frame.phys_addr[0] = hi_mpi_vo_handle_to_phys_addr(vb_pool_val);
    user_frame->v_frame.phys_addr[1] = user_frame->v_frame.phys_addr[0] + luma_size;
    user_frame->v_frame.header_phys_addr[0] = user_frame->v_frame.phys_addr[0];
    user_frame->v_frame.header_phys_addr[1] = user_frame->v_frame.phys_addr[1];
}

hi_void vo_hdmi_init(hi_s32 dev, hi_s32 layer, hi_vo_intf_type intf_type, hi_vo_intf_sync intf_sync, vo_mst_sync_info sync_info)
{
    vo_init_dev(dev, intf_type, intf_sync);
    vo_init_layer(layer, sync_info.height, sync_info.width, sync_info.frame_rate);
    vo_init_chn(layer, sync_info.height, sync_info.width);
    hi_mpi_hdmi_init_sample();
    hi_mpi_hdmi_avi_infoframe_colorspace(HI_HDMI_VIDEO_FORMAT_1080P_60, 148500);
}

hi_void vo_hdmi_deinit(hi_s32 dev, hi_s32 layer)
{
    hi_mpi_vo_disable_chn(layer, 0);
    hi_mpi_vo_disable_video_layer(layer);
    VO_CHECK_RET(hi_mpi_vo_disable(dev), "hi_mpi_vo_disable");

    hi_mpi_hdmi_stop(HI_HDMI_ID_0);
    hi_mpi_hdmi_close(HI_HDMI_ID_0);
    hi_mpi_hdmi_deinit();
}
} // namespace

HdmiOutputThread::HdmiOutputThread(aclrtRunMode runMode, VencConfig vencConfig)
    : runMode_(runMode),
      vencConfig_(vencConfig),
      vbPoolVal_(static_cast<hi_u32>(-1)),
      sysInited_(false),
      hdmiInited_(false),
      devId_(DEV_DHD0),
      layerId_(VO_LAYER_VHD0),
      intfType_(HI_VO_INTF_HDMI),
      intfSync_(HI_VO_OUT_1080P60)
{
    (void)memset(&syncInfo_, 0, sizeof(syncInfo_));
}

HdmiOutputThread::~HdmiOutputThread()
{
    DeinitHdmi();
}

AclLiteError HdmiOutputThread::Init()
{
    uint32_t desiredWidth = vencConfig_.outputWidth;   // 期望输出宽度
    uint32_t desiredHeight = vencConfig_.outputHeight; // 期望输出高度
    uint32_t desiredFps = vencConfig_.outputFps;       // 期望输出帧率
    if (desiredWidth == 0 || desiredHeight == 0)
    {
        desiredWidth = 1920;
        desiredHeight = 1080;
    }
    intfSync_ = GetHdmiIntfSyncBySize(desiredWidth, desiredHeight);
    vo_mipi_get_sync_info(intfSync_, &syncInfo_);
    // 与 VO 时序保持一致，避免参数不匹配
    vencConfig_.outputWidth = syncInfo_.width;
    vencConfig_.outputHeight = syncInfo_.height;
    if (desiredFps == 0 || desiredFps > syncInfo_.frame_rate)
    {
        vencConfig_.outputFps = syncInfo_.frame_rate;
    }
    else
    {
        vencConfig_.outputFps = desiredFps;
    }

    AclLiteError ret = InitHdmi();
    if (ret != ACLLITE_OK) {
        ACLLITE_LOG_ERROR("Init HDMI failed");
        return ret;
    }
    ACLLITE_LOG_INFO("HDMI init done, resolution %ux%u@%u (target_fps=%u)",
                     syncInfo_.width,
                     syncInfo_.height,
                     syncInfo_.frame_rate,
                     vencConfig_.outputFps);
    return ACLLITE_OK;
}

AclLiteError HdmiOutputThread::InitHdmi()
{
    vo_mipi_sys_init(syncInfo_.height, syncInfo_.width);
    sysInited_ = true;

    // 注意：intfSync 需要使用 HI_VO_OUT_1080P60 | HI_VO_INTF_VGA (与 sample_hdmi.c 一致)
    hi_vo_intf_sync actualIntfSync = static_cast<hi_vo_intf_sync>(intfSync_ | HI_VO_INTF_VGA);
    vo_hdmi_init(devId_, layerId_, intfType_, actualIntfSync, syncInfo_);

    // 将 VB 池创建放在 VO/HDMI 初始化之后，和 sample_hdmi 顺序保持一致，避免 VO 未准备好导致帧不可见
    if (vo_mipi_create_vb_pool(syncInfo_.height, syncInfo_.width, &vbPoolVal_) != HI_SUCCESS) {
        ACLLITE_LOG_ERROR("Create VB pool for HDMI failed");
        return ACLLITE_ERROR;
    }

    hdmiInited_ = true;
    ACLLITE_LOG_INFO("vo_hdmi_init completed with intfSync=0x%x", actualIntfSync);
    return ACLLITE_OK;
}

void HdmiOutputThread::DeinitHdmi()
{
    if (hdmiInited_) {
        vo_hdmi_deinit(devId_, layerId_);
        hdmiInited_ = false;
    }
    if (vbPoolVal_ != static_cast<hi_u32>(-1)) {
        hi_mpi_vo_destroy_pool(vbPoolVal_);
        vbPoolVal_ = static_cast<hi_u32>(-1);
    }
    if (sysInited_) {
        vo_mipi_sys_exit();
        sysInited_ = false;
    }
}

AclLiteError HdmiOutputThread::EnsureImageOnHost(const ImageData &deviceImg, ImageData &hostImg)
{
    if (deviceImg.data == nullptr || deviceImg.size == 0) {
        return ACLLITE_ERROR;
    }
    if (runMode_ == ACL_HOST) {
        hostImg = deviceImg;
        return ACLLITE_OK;
    }
    ImageData deviceCopy = deviceImg;
    return CopyImageToLocal(hostImg, deviceCopy, runMode_);
}

AclLiteError HdmiOutputThread::DisplayFrame(const ImageData &image)
{
    if (image.width != syncInfo_.width || image.height != syncInfo_.height) {
        ACLLITE_LOG_ERROR("Image size %ux%u mismatch HDMI %ux%u", image.width, image.height, syncInfo_.width, syncInfo_.height);
        return ACLLITE_ERROR;
    }
    size_t expectedSize = static_cast<size_t>(image.width) * image.height * 3 / 2;
    if (image.size < expectedSize) {
        ACLLITE_LOG_ERROR("Image size %u too small, expected at least %zu", image.size, expectedSize);
        return ACLLITE_ERROR;
    }
    
    // 验证数据指针
    if (image.data == nullptr) {
        return ACLLITE_ERROR;
    }
    
    hi_video_frame_info user_frame;
    (void)memset(&user_frame, 0, sizeof(user_frame));
    vo_init_user_frame(vbPoolVal_, syncInfo_.height, syncInfo_.width, &user_frame);

    hi_u8 *y_plane = reinterpret_cast<hi_u8 *>(static_cast<uintptr_t>(user_frame.v_frame.phys_addr[0]));
    hi_u8 *uv_plane = reinterpret_cast<hi_u8 *>(static_cast<uintptr_t>(user_frame.v_frame.phys_addr[1]));
    
    if (y_plane == nullptr || uv_plane == nullptr) {
        ACLLITE_LOG_ERROR("Invalid VO plane address");
        return ACLLITE_ERROR;
    }
    size_t luma = syncInfo_.width * syncInfo_.height;
    size_t strideY = user_frame.v_frame.width_stride[0];
    size_t strideUV = user_frame.v_frame.width_stride[1];
    
    // 按行拷贝，保证对齐填充区不被跨行写错
    const uint8_t *srcY = image.data.get();
    for (hi_u32 h = 0; h < syncInfo_.height; ++h) {
        (void)memcpy(y_plane + h * strideY, srcY + h * syncInfo_.width, syncInfo_.width);
    }
    
    const uint8_t *srcUV = image.data.get() + luma;
    for (hi_u32 h = 0; h < syncInfo_.height / 2; ++h) {
        (void)memcpy(uv_plane + h * strideUV, srcUV + h * syncInfo_.width, syncInfo_.width);
    }

    // NOTE: send to video layer (not VO device) to ensure the frame shows up.
    // Keep retrying for a short window to avoid dropping decimated frames.
    const int maxRetry = 50; // ~100ms with 2ms sleep
    hi_s32 ret = HI_FAILURE;
    for (int attempt = 0; attempt < maxRetry; ++attempt) {
        ret = hi_mpi_vo_send_frame(layerId_, 0, &user_frame, 0);
        if (ret == HI_SUCCESS) {
            break;
        }
        if (attempt == 0 || (attempt + 1) % 10 == 0) {
            ACLLITE_LOG_WARNING("hi_mpi_vo_send_frame failed (0x%x) attempt %d/%d", ret, attempt + 1, maxRetry);
        }
        usleep(2000); // short backoff to let VO consume previous frame
    }
    
    if (ret != HI_SUCCESS) {
        static int dropCount = 0;
        if (++dropCount % 10 == 0) {
            ACLLITE_LOG_ERROR("hi_mpi_vo_send_frame keep failing, dropped %d frames (last ret=0x%x)", dropCount, ret);
        }
        return ACLLITE_OK;
    }
    return ACLLITE_OK;
}

AclLiteError HdmiOutputThread::HandleDisplay(std::shared_ptr<DetectDataMsg> detectDataMsg)
{
    if (!hdmiInited_) {
        ACLLITE_LOG_ERROR("HDMI is not initialized");
        return ACLLITE_ERROR;
    }
    uint32_t targetFps = vencConfig_.outputFps; // 目标输出帧率
    if (targetFps == 0 || targetFps > syncInfo_.frame_rate)
    {
        targetFps = syncInfo_.frame_rate;
    }
    int64_t frameIntervalUs = 1000000 / targetFps; // 帧间隔微秒
    static auto lastSendTime = std::chrono::steady_clock::time_point(); // 上次发送时间
    static bool hasLastSend = false; // 是否已有发送记录
    for (size_t i = 0; i < detectDataMsg->decodedImg.size(); ++i) {
        if (!detectDataMsg->isLastFrame && hasLastSend)
        {
            auto now = std::chrono::steady_clock::now(); // 当前时间
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                now - lastSendTime).count();
            if (elapsed < frameIntervalUs)
            {
                continue;
            }
        }
        ImageData hostImg;
        AclLiteError ret = EnsureImageOnHost(detectDataMsg->decodedImg[i], hostImg);
        if (ret != ACLLITE_OK) {
            ACLLITE_LOG_ERROR("Copy image to host for HDMI failed, error %d", ret);
            return ret;
        }
        ret = DisplayFrame(hostImg);
        if (ret != ACLLITE_OK) {
            ACLLITE_LOG_ERROR("Display frame to HDMI failed, error %d", ret);
            return ret;
        }
        lastSendTime = std::chrono::steady_clock::now();
        hasLastSend = true;
    }

    if (detectDataMsg->isLastFrame) {
        DeinitHdmi();
        SendMessage(g_MainThreadId, MSG_APP_EXIT, nullptr);
    }
    return ACLLITE_OK;
}

AclLiteError HdmiOutputThread::Process(int msgId, std::shared_ptr<void> msgData)
{
    auto start = std::chrono::high_resolution_clock::now();
    AclLiteError ret = ACLLITE_OK;
    switch (msgId) {
    case MSG_HDMI_DISPLAY:
        ret = HandleDisplay(std::static_pointer_cast<DetectDataMsg>(msgData));
        break;
    case MSG_ENCODE_FINISH:
        DeinitHdmi();
        SendMessage(g_MainThreadId, MSG_APP_EXIT, nullptr);
        break;
    default:
        ACLLITE_LOG_INFO("HDMI thread ignore msg %d", msgId);
        break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (msgId == MSG_HDMI_DISPLAY) {
        static int logCount = 0;
        if (++logCount % 30 == 0) {
            ACLLITE_LOG_INFO("[HdmiOutputThread] Process time: %ld ms", duration);
        }
    }
    return ret;
}
