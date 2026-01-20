#include "tracking.h"
#include "AclLiteApp.h"
#include "AclLiteUtils.h"
#include "Params.h"
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>

// Constants for normalization
const float Tracking::mean_vals[3] = {
    0.485f * 255,
    0.456f * 255,
    0.406f * 255};
const float Tracking::norm_vals[3] = {
    1.0f / 0.229f / 255.f,
    1.0f / 0.224f / 255.f,
    1.0f / 0.225f / 255.f};

namespace
{
constexpr const char *kDefaultMixFormerModel =
    "model/mixformerv2_online_small.om";

const uint32_t kSleepTime = 500;

// Model input/output tensor names (these should match your OM model)
constexpr const char *kInputTemplateName = "template";
constexpr const char *kInputOnlineTemplateName = "online_template";
constexpr const char *kInputSearchName = "search";
constexpr const char *kOutputBoxesName = "pred_boxes";
constexpr const char *kOutputScoresName = "pred_scores";
} // namespace

Tracking::Tracking(const std::string &model_path)
    : template_size(112),
      search_size(224),
      template_factor(2.0f),
      search_factor(5.0f),
      frame_id(0),
      max_pred_score(0.0f),
      update_interval(200),
      template_update_score_threshold(0.85f),
      max_score_decay(0.98f),
      model_path_(model_path.empty() ? kDefaultMixFormerModel : model_path),
      model_initialized_(false)
{
    // Initialize buffer sizes based on model dimensions
    // template: [1, 3, 112, 112] = 37632
    input_template_size = 1 * 3 * 112 * 112;
    input_online_template_size = 1 * 3 * 112 * 112;
    // search: [1, 3, 224, 224] = 150528
    input_search_size = 1 * 3 * 224 * 224;
    // pred_boxes: [1, 1, 4] = 4
    output_pred_boxes_size = 1 * 1 * 4;
    // pred_scores: [1] = 1
    output_pred_scores_size = 1;

    // Allocate host memory
    input_template = new float[input_template_size];
    input_online_template = new float[input_online_template_size];
    input_search = new float[input_search_size];
    output_pred_boxes = new float[output_pred_boxes_size];
    output_pred_scores = new float[output_pred_scores_size];

    // Initialize to zero
    std::memset(input_template, 0, input_template_size * sizeof(float));
    std::memset(
        input_online_template,
        0,
        input_online_template_size * sizeof(float));
    std::memset(input_search, 0, input_search_size * sizeof(float));
    std::memset(output_pred_boxes, 0, output_pred_boxes_size * sizeof(float));
    std::memset(output_pred_scores, 0, output_pred_scores_size * sizeof(float));

    object_box = {{0, 0, 0, 0, 0, 0, 0, 0}, 0.0f, 0};
    resetMaxPredScore();
}

Tracking::~Tracking()
{
    if (model_initialized_)
    {
        model_.DestroyResource();
    }

    delete[] output_pred_boxes;
    delete[] output_pred_scores;
    delete[] input_template;
    delete[] input_online_template;
    delete[] input_search;
}

int Tracking::InitModel()
{
    if (model_initialized_)
    {
        ACLLITE_LOG_WARNING("Model already initialized");
        return 0;
    }

    // Check if model file exists (optional validation)
    std::ifstream modelFile(model_path_);
    if (!modelFile.good())
    {
        ACLLITE_LOG_WARNING("Model file not accessible: %s, attempting to load anyway", model_path_.c_str());
    }

    ACLLITE_LOG_INFO("Tracking OM initializing with model path: %s", model_path_.c_str());
    AclLiteError ret = model_.Init(model_path_);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Tracking OM model init failed for path [%s], error: %d", model_path_.c_str(), ret);
        return -1;
    }

    model_initialized_ = true;
    ACLLITE_LOG_INFO("Tracking OM model initialized successfully from: %s", model_path_.c_str());
    return 0;
}

// AclLiteThread lifecycle
AclLiteError Tracking::Init()
{
    // Acquire run mode from current context
    aclError aclRet = aclrtGetRunMode(&runMode_);
    if (aclRet != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Get run mode failed in tracking thread Init");
        return ACLLITE_ERROR; // generic error
    }

    // Initialize OM model if needed
    if (InitModel() != 0)
    {
        return ACLLITE_ERROR; // model init failed
    }

    return ACLLITE_OK;
}

void Tracking::SendTrackingStateFeedback(std::shared_ptr<DetectDataMsg> detectDataMsg)
{
    if (dataInputThreadId_ < 0)
    {
        // No data input thread to send feedback to
        return;
    }

    std::shared_ptr<DetectDataMsg> feedbackMsg = std::make_shared<DetectDataMsg>();
    feedbackMsg->trackingActive = detectDataMsg->trackingActive;
    feedbackMsg->trackingConfidence = detectDataMsg->trackingConfidence;
    feedbackMsg->needRedetection = detectDataMsg->needRedetection;
    feedbackMsg->channelId = detectDataMsg->channelId;
    feedbackMsg->filterStaticTargetEnabled = filter_static_target_;
    feedbackMsg->hasBlockedTarget = has_blocked_target_;
    feedbackMsg->blockedCenterX = blocked_target_.cx;
    feedbackMsg->blockedCenterY = blocked_target_.cy;
    feedbackMsg->blockedWidth = blocked_target_.w;
    feedbackMsg->blockedHeight = blocked_target_.h;
    feedbackMsg->staticCenterThreshold = static_center_threshold_;
    feedbackMsg->staticSizeThreshold = static_size_threshold_;
    
    AclLiteError ret = SendMessage(dataInputThreadId_, MSG_TRACK_STATE_CHANGE, feedbackMsg);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_WARNING("[Tracking Ch%d] Failed to send track state change to DataInput, error %d",
                            detectDataMsg->channelId, ret);
    }
}

bool Tracking::IsBlockedDetection(const DetectionOBB &det) const
{
    if (!has_blocked_target_)
    {
        return false;
    }
    float det_w = det.x1 - det.x0;
    float det_h = det.y1 - det.y0;
    float det_cx = (det.x0 + det.x1) * 0.5f;
    float det_cy = (det.y0 + det.y1) * 0.5f;

    bool center_match =
        std::fabs(det_cx - blocked_target_.cx) <= static_center_threshold_ &&
        std::fabs(det_cy - blocked_target_.cy) <= static_center_threshold_;
    bool size_match =
        std::fabs(det_w - blocked_target_.w) <= static_size_threshold_ &&
        std::fabs(det_h - blocked_target_.h) <= static_size_threshold_;

    return center_match && size_match;
}

bool Tracking::UpdateStaticTrackingState(const DrBBox &box)
{
    if (!filter_static_target_ || box.w <= 0.0f || box.h <= 0.0f)
    {
        return false;
    }

    if (!has_last_box_)
    {
        last_box_ = box;
        has_last_box_ = true;
        static_frame_count_ = 0;
        return false;
    }

    bool center_stable =
        std::fabs(box.cx - last_box_.cx) <= static_center_threshold_ &&
        std::fabs(box.cy - last_box_.cy) <= static_center_threshold_;
    bool size_stable =
        std::fabs(box.w - last_box_.w) <= static_size_threshold_ &&
        std::fabs(box.h - last_box_.h) <= static_size_threshold_;

    if (center_stable && size_stable)
    {
        static_frame_count_++;
    }
    else
    {
        static_frame_count_ = 0;
    }

    last_box_ = box;
    return static_frame_count_ >= static_frame_threshold_;
}

void Tracking::FillStaticFilterState(std::shared_ptr<DetectDataMsg> &msg) const
{
    msg->filterStaticTargetEnabled = filter_static_target_;
    msg->hasBlockedTarget = has_blocked_target_;
    msg->blockedCenterX = blocked_target_.cx;
    msg->blockedCenterY = blocked_target_.cy;
    msg->blockedWidth = blocked_target_.w;
    msg->blockedHeight = blocked_target_.h;
    msg->staticCenterThreshold = static_center_threshold_;
    msg->staticSizeThreshold = static_size_threshold_;
}

AclLiteError Tracking::Process(int msgId, std::shared_ptr<void> data)
{
    switch (msgId)
    {
    case MSG_TRACK_DATA: // receive detection results from postprocess
    {
        std::shared_ptr<DetectDataMsg> detectDataMsg =
            std::static_pointer_cast<DetectDataMsg>(data);
        // Ensure output thread id cached
        if (dataOutputThreadId_ < 0)
        {
            dataOutputThreadId_ = detectDataMsg->dataOutputThreadId;
        }
        if (dataInputThreadId_ < 0)
        {
            dataInputThreadId_ = detectDataMsg->dataInputThreadId;
        }

        // 单目标跟踪：首次检测初始化，后续调用 track 更新
        if (!detectDataMsg->frame.empty())
        {
            cv::Mat &img = detectDataMsg->frame[0];

            if (!tracking_initialized_)
            {
                // 选择最高分目标作为跟踪目标(后处理已把最佳目标放在首位)
                if (!detectDataMsg->detections.empty())
                {
                    const DetectionOBB *bestPtr = nullptr;
                    for (size_t i = 0; i < detectDataMsg->detections.size(); ++i)
                    {
                        if (!IsBlockedDetection(detectDataMsg->detections[i]))
                        {
                            bestPtr = &detectDataMsg->detections[i];
                            break;
                        }
                    }
                    if (bestPtr == nullptr)
                    {
                        ACLLITE_LOG_INFO(
                            "[Tracking Ch%d] Skip init due to blocked target signature",
                            detectDataMsg->channelId);
                    }
                    else
                    {
                        const DetectionOBB &best = *bestPtr;

                        DrOBB initBox;
                        initBox.box.x0 = best.x0;
                        initBox.box.y0 = best.y0;
                        initBox.box.x1 = best.x1;
                        initBox.box.y1 = best.y1;
                        initBox.class_id = best.class_id;
                        // preserve detection init score in DrOBB for follow-up tracking
                        initBox.score = best.score;
                        initBox.initScore = best.score;

                        if (this->init(img, initBox) == 0)
                        {
                            tracking_initialized_ = true;
                            track_loss_count_ = 0;
                            current_tracking_confidence_ = best.score;
                            static_frame_count_ = 0;
                            last_box_ = initBox.box;
                            has_last_box_ = true;
                            if (has_blocked_target_ &&
                                !IsBlockedDetection(best))
                            {
                                has_blocked_target_ = false;
                            }
                            
                            // Store tracking result in new structure
                            detectDataMsg->trackingResult.bbox = best;
                            detectDataMsg->trackingResult.isTracked = true;
                            detectDataMsg->trackingResult.initScore = best.score;
                            detectDataMsg->trackingResult.curScore = best.score;
                            
                            // 设置跟踪状态
                            detectDataMsg->trackingActive = true;
                            detectDataMsg->trackingConfidence = best.score;
                            detectDataMsg->needRedetection = false;
                            
                            // Keep old fields for backward compatibility
                            detectDataMsg->hasTracking = true;
                            detectDataMsg->trackInitScore = best.score;
                            detectDataMsg->trackScore = best.score;
                            
                            // 通知DataInput跟踪已激活，DataInput可以进入TRACK_ONLY模式
                            SendTrackingStateFeedback(detectDataMsg);
                        }
                    }
                }
            }
            else
            {
                // 已初始化,执行跟踪更新
                const DrOBB &tracked = this->track(img);
                current_tracking_confidence_ = tracked.score;
                
                // Store tracking result in new structure
                detectDataMsg->trackingResult.bbox.x0 = tracked.box.x0;
                detectDataMsg->trackingResult.bbox.y0 = tracked.box.y0;
                detectDataMsg->trackingResult.bbox.x1 = tracked.box.x1;
                detectDataMsg->trackingResult.bbox.y1 = tracked.box.y1;
                detectDataMsg->trackingResult.bbox.score = tracked.score;
                detectDataMsg->trackingResult.bbox.class_id = tracked.class_id;
                detectDataMsg->trackingResult.isTracked = true;
                detectDataMsg->trackingResult.curScore = tracked.score;
                // preserve initial detection confidence across frames
                detectDataMsg->trackingResult.initScore = tracked.initScore;
                
                // 更新跟踪状态
                detectDataMsg->trackingActive = true;
                detectDataMsg->trackingConfidence = tracked.score;
                detectDataMsg->needRedetection = false;
                track_loss_count_ = 0;
                
                // Keep old fields for backward compatibility
                detectDataMsg->hasTracking = true;
                detectDataMsg->trackScore = tracked.score;
                // keep backward compat init score
                detectDataMsg->trackInitScore = tracked.initScore;
            }
        }

        FillStaticFilterState(detectDataMsg);
        // Forward message to DataOutputThread
        MsgSend(detectDataMsg);
        return ACLLITE_OK;
    }
    
    case MSG_TRACK_ONLY: // tracking only (no detection, from DataInput)
    {
        std::shared_ptr<DetectDataMsg> detectDataMsg =
            std::static_pointer_cast<DetectDataMsg>(data);
        
        // Ensure thread ids cached
        if (dataOutputThreadId_ < 0)
        {
            dataOutputThreadId_ = detectDataMsg->dataOutputThreadId;
        }
        if (dataInputThreadId_ < 0)
        {
            dataInputThreadId_ = detectDataMsg->dataInputThreadId;
        }
        
        if (!tracking_initialized_)
        {
            // 跟踪未初始化,需要检测 - 只在首次发送状态反馈
            static bool hasRequestedDetection = false;
            if (!hasRequestedDetection)
            {
                ACLLITE_LOG_WARNING("[Tracking Ch%d] Not initialized, requesting detection",
                                    detectDataMsg->channelId);
                detectDataMsg->trackingActive = false;
                detectDataMsg->needRedetection = true;
                detectDataMsg->trackingConfidence = 0.0f;
                
                // 反馈状态给DataInput
                SendTrackingStateFeedback(detectDataMsg);
                hasRequestedDetection = true;
            }
            
            // 仍然发送到输出(显示原图)
            MsgSend(detectDataMsg);
            return ACLLITE_OK;
        }
        
        // 已初始化，重置标志（用于下次跟踪丢失）
        {
            static bool hasRequestedDetection = false;
            hasRequestedDetection = false;
        }
        
        // 执行跟踪
        if (!detectDataMsg->frame.empty())
        {
            cv::Mat &img = detectDataMsg->frame[0];
            const DrOBB &tracked = this->track(img);
            
            // 更新置信度
            current_tracking_confidence_ = tracked.score;
            
            // Store tracking result
            detectDataMsg->trackingResult.bbox.x0 = tracked.box.x0;
            detectDataMsg->trackingResult.bbox.y0 = tracked.box.y0;
            detectDataMsg->trackingResult.bbox.x1 = tracked.box.x1;
            detectDataMsg->trackingResult.bbox.y1 = tracked.box.y1;
            detectDataMsg->trackingResult.bbox.score = tracked.score;
            detectDataMsg->trackingResult.bbox.class_id = tracked.class_id;
            detectDataMsg->trackingResult.isTracked = true;
            detectDataMsg->trackingResult.curScore = tracked.score;
            // preserve initial detection confidence across frames
            detectDataMsg->trackingResult.initScore = tracked.initScore;
            
            // Keep old fields
            detectDataMsg->hasTracking = true;
            detectDataMsg->trackScore = tracked.score;
            // keep backward compat init score
            detectDataMsg->trackInitScore = tracked.initScore;
            
            // ============ 判断是否需要重新检测 ============
            bool needRedetection = false;
            bool isStaticSuspect = false;
            if (filter_static_target_)
            {
                isStaticSuspect = UpdateStaticTrackingState(tracked.box);
            }
            
            if (tracked.score < confidence_redetect_threshold_)
            {
                track_loss_count_++;
                ACLLITE_LOG_WARNING("[Tracking Ch%d Frame%d] Low confidence: %.3f (threshold=%.3f), loss_count=%d",
                                    detectDataMsg->channelId, detectDataMsg->msgNum,
                                    tracked.score, confidence_redetect_threshold_, track_loss_count_);
            }
            else
            {
                track_loss_count_ = 0;
            }
            
            if (track_loss_count_ >= max_track_loss_frames_)
            {
                needRedetection = true;
                ACLLITE_LOG_INFO("[Tracking Ch%d] Lost tracking after %d frames, requesting redetection",
                                 detectDataMsg->channelId, track_loss_count_);
                tracking_initialized_ = false;  // 重置跟踪状态
                track_loss_count_ = 0;
            }
            
            if (isStaticSuspect)
            {
                needRedetection = true;
                tracking_initialized_ = false;
                track_loss_count_ = 0;
                static_frame_count_ = 0;
                has_last_box_ = false;
                has_blocked_target_ = true;
                blocked_target_ = tracked.box;
                detectDataMsg->trackingResult.isTracked = false;
                detectDataMsg->hasTracking = false;
                ACLLITE_LOG_WARNING(
                    "[Tracking Ch%d] Suspect static target, block center(%.1f,%.1f) size(%.1f,%.1f)",
                    detectDataMsg->channelId,
                    blocked_target_.cx, blocked_target_.cy,
                    blocked_target_.w, blocked_target_.h);
            }

            // 更新消息状态
            detectDataMsg->trackingActive = !needRedetection && (tracked.score >= confidence_active_threshold_);
            detectDataMsg->trackingConfidence = tracked.score;
            detectDataMsg->needRedetection = needRedetection;
            
            // 如果需要重新检测,反馈给DataInput
            if (needRedetection)
            {
                SendTrackingStateFeedback(detectDataMsg);
            }
        }
        
        FillStaticFilterState(detectDataMsg);
        // Forward to output
        MsgSend(detectDataMsg);
        return ACLLITE_OK;
    }
    
    default:
        ACLLITE_LOG_INFO("Tracking thread ignore msg %d", msgId);
        return ACLLITE_OK;
    }
}

AclLiteError Tracking::MsgSend(std::shared_ptr<DetectDataMsg> detectDataMsg)
{
    while (1)
    {
        AclLiteError ret = SendMessage(dataOutputThreadId_, MSG_OUTPUT_FRAME, detectDataMsg);
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
            ACLLITE_LOG_ERROR("Tracking send output frame message failed, error %d", ret);
            return ret;
        }
    }

    if (detectDataMsg->isLastFrame)
    {
        while (1)
        {
            AclLiteError ret = SendMessage(
                dataOutputThreadId_,
                MSG_ENCODE_FINISH,
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
                ACLLITE_LOG_ERROR(
                    "Tracking send encode finish message failed, error %d",
                    ret);
                return ret;
            }
        }
    }

    return ACLLITE_OK;
}

void Tracking::setTemplateSize(int size)
{
    if (size > 0)
    {
        this->template_size = size;
    }
}

void Tracking::setSearchSize(int size)
{
    if (size > 0)
    {
        this->search_size = size;
    }
}

void Tracking::setTemplateFactor(float factor)
{
    if (factor > 0.0f)
    {
        this->template_factor = factor;
    }
}

void Tracking::setSearchFactor(float factor)
{
    if (factor > 0.0f)
    {
        this->search_factor = factor;
    }
}

void Tracking::setUpdateInterval(int interval)
{
    this->update_interval = interval;
}

void Tracking::setTemplateUpdateScoreThreshold(float threshold)
{
    this->template_update_score_threshold = threshold;
}

void Tracking::setMaxScoreDecay(float decay)
{
    this->max_score_decay = decay;
}

void Tracking::setConfidenceActiveThreshold(float threshold)
{
    this->confidence_active_threshold_ = threshold;
}

void Tracking::setConfidenceRedetectThreshold(float threshold)
{
    this->confidence_redetect_threshold_ = threshold;
}

void Tracking::setMaxTrackLossFrames(int maxFrames)
{
    this->max_track_loss_frames_ = maxFrames;
}

void Tracking::setStaticTargetFilterEnabled(bool enabled)
{
    this->filter_static_target_ = enabled;
}

void Tracking::setStaticCenterThreshold(float threshold)
{
    if (threshold > 0.0f)
    {
        this->static_center_threshold_ = threshold;
    }
}

void Tracking::setStaticSizeThreshold(float threshold)
{
    if (threshold > 0.0f)
    {
        this->static_size_threshold_ = threshold;
    }
}

void Tracking::setStaticFrameThreshold(int frames)
{
    if (frames > 0)
    {
        this->static_frame_threshold_ = frames;
    }
}

void Tracking::resetMaxPredScore() { this->max_pred_score = 0.0f; }

int Tracking::init(const cv::Mat &img, DrOBB bbox)
{
    if (!model_initialized_)
    {
        ACLLITE_LOG_ERROR("Model not initialized, call InitModel() first");
        return -1;
    }

    cv::Mat template_patch;
    float   resize_factor = 1.f;

    bbox.box.w = bbox.box.x1 - bbox.box.x0;
    bbox.box.h = bbox.box.y1 - bbox.box.y0;
    bbox.box.cx = bbox.box.x0 + 0.5f * bbox.box.w;
    bbox.box.cy = bbox.box.y0 + 0.5f * bbox.box.h;

    int ret = sample_target(
        img,
        template_patch,
        bbox.box,
        this->template_factor,
        this->template_size,
        resize_factor);
    if (ret != 0)
    {
        ACLLITE_LOG_ERROR("Failed to sample template patch");
        return -1;
    }

    half_norm(template_patch, this->input_template);
    std::memcpy(
        this->input_online_template,
        this->input_template,
        this->input_template_size * sizeof(float));

    this->state = bbox.box;
    this->object_box.box = bbox.box;
    // set initial tracking confidence from detection score if available
    this->object_box.score = bbox.score > 0 ? bbox.score : 1.0f;
    this->object_box.initScore = bbox.initScore > 0 ? bbox.initScore : bbox.score;
    this->object_box.class_id = bbox.class_id;
    this->resetMaxPredScore();
    this->frame_id = 0;

    return 0;
}

const DrOBB &Tracking::track(const cv::Mat &img)
{
    if (!model_initialized_)
    {
        ACLLITE_LOG_ERROR("Model not initialized");
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    const int interval =
        this->update_interval > 0 ? this->update_interval : 200;
    const int current_frame_id = this->frame_id;
    if (this->frame_id >= std::numeric_limits<int>::max())
    {
        this->frame_id = 0;
    }
    else
    {
        ++this->frame_id;
    }

    this->max_pred_score *= this->max_score_decay;

    cv::Mat search_patch;
    float   search_resize_factor = 1.f;
    int     ret = sample_target(
        img,
        search_patch,
        this->state,
        this->search_factor,
        this->search_size,
        search_resize_factor);
    if (ret != 0)
    {
        ACLLITE_LOG_ERROR("Failed to sample search patch");
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    half_norm(search_patch, this->input_search);

    bool    has_online_template_patch = false;
    cv::Mat online_template_patch;
    float   template_resize_factor = 1.f;
    ret = sample_target(
        img,
        online_template_patch,
        this->state,
        this->template_factor,
        this->template_size,
        template_resize_factor);
    if (ret == 0)
    {
        has_online_template_patch = true;
    }

    infer();

    DrBBox pred_box = this->cal_bbox(
        this->output_pred_boxes,
        search_resize_factor,
        this->search_size);
    float pred_score =
        this->output_pred_scores_size > 0 ? this->output_pred_scores[0] : 0.f;

    if (pred_box.w <= 0.f || pred_box.h <= 0.f)
    {
        ACLLITE_LOG_WARNING("Invalid prediction box");
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    this->map_box_back(pred_box, search_resize_factor, this->search_size);
    this->clip_box(pred_box, img.rows, img.cols, 0);

    this->state = pred_box;
    this->object_box.box = pred_box;
    this->object_box.score = pred_score;

    const bool should_update_online_template =
        (current_frame_id % interval == 0);

    const bool can_refresh_online_template =
        has_online_template_patch &&
        pred_score > this->template_update_score_threshold &&
        pred_score > this->max_pred_score;

    if (can_refresh_online_template)
    {
        if (this->new_online_template.size() !=
            static_cast<size_t>(this->input_online_template_size))
        {
            this->new_online_template.resize(this->input_online_template_size);
        }

        half_norm(online_template_patch, this->new_online_template.data());
        this->max_pred_score = pred_score;
    }

    if (should_update_online_template &&
        this->new_online_template.size() ==
            static_cast<size_t>(this->input_online_template_size))
    {
        std::memcpy(
            this->input_online_template,
            this->new_online_template.data(),
            this->input_online_template_size * sizeof(float));
    }

    return this->object_box;
}

void Tracking::infer()
{
    if (!model_initialized_)
    {
        ACLLITE_LOG_ERROR("Model not initialized");
        return;
    }

    // Prepare input data vector for multiple inputs
    std::vector<DataInfo> inputData;
    DataInfo              template_input;
    template_input.data = input_template;
    template_input.size = input_template_size * sizeof(float);
    inputData.push_back(template_input);

    DataInfo online_template_input;
    online_template_input.data = input_online_template;
    online_template_input.size = input_online_template_size * sizeof(float);
    inputData.push_back(online_template_input);

    DataInfo search_input;
    search_input.data = input_search;
    search_input.size = input_search_size * sizeof(float);
    inputData.push_back(search_input);

    // Create input dataset
    AclLiteError ret = model_.CreateInput(inputData);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Create model input dataset failed, error: %d", ret);
        return;
    }

    // Execute inference
    std::vector<InferenceOutput> inferenceOutput;
    ret = model_.ExecuteV2(inferenceOutput);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Execute model inference failed, error: %d", ret);
        model_.DestroyInput();
        return;
    }

    // Extract output data
    if (inferenceOutput.size() >= 2)
    {
        aclrtRunMode runMode;
        aclError     aclRet = aclrtGetRunMode(&runMode);
        if (aclRet != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("Get run mode failed");
            model_.DestroyInput();
            return;
        }

        if (inferenceOutput[0].size >= output_pred_boxes_size * sizeof(float))
        {
            void *hostBuffer = CopyDataToHost(
                inferenceOutput[0].data.get(),
                output_pred_boxes_size * sizeof(float),
                runMode,
                MEMORY_NORMAL);
            if (hostBuffer != nullptr)
            {
                std::memcpy(
                    output_pred_boxes,
                    hostBuffer,
                    output_pred_boxes_size * sizeof(float));
                delete[] static_cast<uint8_t *>(hostBuffer);
            }
            else
            {
                ACLLITE_LOG_ERROR("Copy boxes output to host failed");
            }
        }
        else
        {
            ACLLITE_LOG_ERROR(
                "Output boxes size mismatch: expected %zu, got %u",
                output_pred_boxes_size * sizeof(float),
                inferenceOutput[0].size);
        }

        // Copy pred_scores output from device to host
        if (inferenceOutput[1].size >= output_pred_scores_size * sizeof(float))
        {
            void *hostBuffer = CopyDataToHost(
                inferenceOutput[1].data.get(),
                output_pred_scores_size * sizeof(float),
                runMode,
                MEMORY_NORMAL);
            if (hostBuffer != nullptr)
            {
                std::memcpy(
                    output_pred_scores,
                    hostBuffer,
                    output_pred_scores_size * sizeof(float));
                delete[] static_cast<uint8_t *>(hostBuffer);
            }
            else
            {
                ACLLITE_LOG_ERROR("Copy scores output to host failed");
            }
        }
        else
        {
            ACLLITE_LOG_ERROR(
                "Output scores size mismatch: expected %zu, got %u",
                output_pred_scores_size * sizeof(float),
                inferenceOutput[1].size);
        }
    }
    else
    {
        ACLLITE_LOG_ERROR(
            "Invalid number of outputs: expected 2, got %zu",
            inferenceOutput.size());
    }

    // Clean up input
    model_.DestroyInput();
}

// -------- Tracking utilities (simplified adapters) --------
int Tracking::sample_target(
    const cv::Mat &img,
    cv::Mat       &patch,
    const DrBBox  &bbox,
    float          factor,
    int            output_size,
    float         &resize_factor)
{
    if (bbox.w <= 0 || bbox.h <= 0 || bbox.cx <= 0 || bbox.cy <= 0)
    {
        std::cout << "bbox is out of range" << std::endl;
        return -1;
    }

    int w = bbox.w;
    int h = bbox.h;
    int crop_sz = std::ceil(std::sqrt(w * h) * factor);

    float cx = bbox.cx;
    float cy = bbox.cy;
    int   x1 = std::round(cx - crop_sz * 0.5);
    int   y1 = std::round(cy - crop_sz * 0.5);

    int x2 = x1 + crop_sz;
    int y2 = y1 + crop_sz;

    int x1_pad = std::max(0, -x1);
    int x2_pad = std::max(x2 - img.cols + 1, 0);

    int y1_pad = std::max(0, -y1);
    int y2_pad = std::max(y2 - img.rows + 1, 0);

    // Crop target
    cv::Rect roi_rect(
        x1 + x1_pad,
        y1 + y1_pad,
        (x2 - x2_pad) - (x1 + x1_pad),
        (y2 - y2_pad) - (y1 + y1_pad));
    if (roi_rect.x < 0 || roi_rect.y < 0 || roi_rect.width <= 0 ||
        roi_rect.height <= 0)
    {
        std::cout << "roi_rect is out of range" << std::endl;
        return -1;
    }
    cv::Mat roi = img(roi_rect);

    // Pad
    cv::copyMakeBorder(
        roi,
        patch,
        y1_pad,
        y2_pad,
        x1_pad,
        x2_pad,
        cv::BORDER_CONSTANT);

    // Resize
    cv::resize(patch, patch, cv::Size(output_size, output_size));

    resize_factor = output_size * 1.f / crop_sz;

    return 0;
}

void Tracking::half_norm(const cv::Mat &img, float *output)
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;

    cv::Mat img_cp;
    img_cp = img.clone();
    cvtColor(img_cp, img_cp, cv::COLOR_BGR2RGB);

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                output[c * img_w * img_h + h * img_w + w] =
                    cv::saturate_cast<float>(
                        (((float)img_cp.at<cv::Vec3b>(h, w)[c]) -
                         mean_vals[c]) *
                        norm_vals[c]);
            }
        }
    }
}

DrBBox Tracking::cal_bbox(
    const float *pred_boxes,
    float        resize_factor,
    int          search_size)
{
    DrBBox pred_box = {0, 0, 0, 0, 0, 0, 0, 0};

    float cx = pred_boxes[0];
    float cy = pred_boxes[1];
    float w = pred_boxes[2];
    float h = pred_boxes[3];

    if (cx < 0 || cy < 0 || w <= 0 || h <= 0)
    {
        return pred_box;
    }

    cx = cx * search_size / resize_factor;
    cy = cy * search_size / resize_factor;
    w = w * search_size / resize_factor;
    h = h * search_size / resize_factor;

    pred_box.x0 = cx - 0.5 * w;
    pred_box.y0 = cy - 0.5 * h;
    pred_box.x1 = pred_box.x0 + w;
    pred_box.y1 = pred_box.y0 + h;
    pred_box.w = w;
    pred_box.h = h;
    pred_box.cx = cx;
    pred_box.cy = cy;

    return pred_box;
}

void Tracking::map_box_back(
    DrBBox &pred_box,
    float   resize_factor,
    int     search_size)
{
    float cx_prev = this->state.cx;
    float cy_prev = this->state.cy;

    float half_side = 0.5 * search_size / resize_factor;

    float w = pred_box.w;
    float h = pred_box.h;
    float cx = pred_box.cx;
    float cy = pred_box.cy;

    float cx_real = cx + (cx_prev - half_side);
    float cy_real = cy + (cy_prev - half_side);

    pred_box.x0 = cx_real - 0.5 * w;
    pred_box.y0 = cy_real - 0.5 * h;
    pred_box.x1 = cx_real + 0.5 * w;
    pred_box.y1 = cy_real + 0.5 * h;
    pred_box.w = w;
    pred_box.h = h;
    pred_box.cx = cx_real;
    pred_box.cy = cy_real;
}

void Tracking::clip_box(DrBBox &box, int img_h, int img_w, int border)
{
    box.x0 = std::min(std::max(0, int(box.x0)), img_w - border);
    box.y0 = std::min(std::max(0, int(box.y0)), img_h - border);
    box.x1 = std::min(std::max(border, int(box.x1)), img_w);
    box.y1 = std::min(std::max(border, int(box.y1)), img_h);
}

void Tracking::setInputTemplateData(const float *data, size_t size)
{
    if (data && size == input_template_size)
    {
        std::memcpy(input_template, data, size * sizeof(float));
    }
}

void Tracking::setInputOnlineTemplateData(const float *data, size_t size)
{
    if (data && size == input_online_template_size)
    {
        std::memcpy(input_online_template, data, size * sizeof(float));
    }
}

void Tracking::setInputSearchData(const float *data, size_t size)
{
    if (data && size == input_search_size)
    {
        std::memcpy(input_search, data, size * sizeof(float));
    }
}

std::vector<float> Tracking::getOutputPredBoxes() const
{
    std::vector<float> result(output_pred_boxes,
                               output_pred_boxes + output_pred_boxes_size);
    return result;
}

std::vector<float> Tracking::getOutputPredScores() const
{
    std::vector<float> result(output_pred_scores,
                               output_pred_scores + output_pred_scores_size);
    return result;
}
