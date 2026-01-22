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
#include <algorithm>

namespace
{
constexpr const char *kDefaultNanotrackHeadModel =
    "model/nanotrack_head_bs1.om";
constexpr const char *kDefaultNanotrackBackboneModel =
    "model/nanotrack_backbone_bs1.om";
constexpr const char *kDefaultNanotrackBackboneSearchModel =
    "model/nanotrack_backbone_search_bs1.om";

const uint32_t kSleepTime = 500;
} // namespace

Tracking::Tracking(const std::string &model_path)
    : frame_id(0),
      update_interval(200),
      template_update_score_threshold(0.85f),
      max_score_decay(0.98f),
      model_initialized_(false)
{
    InitNanotrackModelPath(model_path);
    EnsureScoreSize(this->cfg_.score_size);
    object_box = {{0, 0, 0, 0, 0, 0, 0, 0}, 0.0f, 0};
}

Tracking::~Tracking()
{
    if (model_initialized_)
    {
        head_model_.DestroyResource();
        backbone_model_.DestroyResource();
        if (has_search_backbone_)
        {
            search_model_.DestroyResource();
        }
    }
}

int Tracking::InitModel()
{
    if (model_initialized_)
    {
        ACLLITE_LOG_WARNING("Model already initialized");
        return 0;
    }

    if (head_model_path_.empty() || backbone_model_path_.empty())
    {
        ACLLITE_LOG_ERROR("Nanotrack model path not initialized");
        return -1;
    }

    std::ifstream headFile(head_model_path_);
    if (!headFile.good())
    {
        ACLLITE_LOG_WARNING("Head model file not accessible: %s, attempting to load anyway",
                            head_model_path_.c_str());
    }
    std::ifstream backboneFile(backbone_model_path_);
    if (!backboneFile.good())
    {
        ACLLITE_LOG_WARNING("Backbone model file not accessible: %s, attempting to load anyway",
                            backbone_model_path_.c_str());
    }
    if (!search_model_path_.empty())
    {
        std::ifstream searchFile(search_model_path_);
        if (!searchFile.good())
        {
            ACLLITE_LOG_WARNING("Search backbone model file not accessible: %s, attempting to load anyway",
                                search_model_path_.c_str());
        }
    }

    ACLLITE_LOG_INFO("Nanotrack OM initializing with head: %s", head_model_path_.c_str());
    AclLiteError ret = head_model_.Init(head_model_path_);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Head model init failed for path [%s], error: %d",
                          head_model_path_.c_str(), ret);
        return -1;
    }

    ACLLITE_LOG_INFO("Nanotrack OM initializing with backbone: %s", backbone_model_path_.c_str());
    ret = backbone_model_.Init(backbone_model_path_);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Backbone model init failed for path [%s], error: %d",
                          backbone_model_path_.c_str(), ret);
        return -1;
    }

    has_search_backbone_ = false;
    if (!search_model_path_.empty() && search_model_path_ != backbone_model_path_)
    {
        ACLLITE_LOG_INFO("Nanotrack OM initializing with search backbone: %s",
                         search_model_path_.c_str());
        ret = search_model_.Init(search_model_path_);
        if (ret == ACLLITE_OK)
        {
            has_search_backbone_ = true;
        }
        else
        {
            ACLLITE_LOG_WARNING("Search backbone init failed, fallback to backbone, error: %d",
                                ret);
        }
    }

    if (InitNanotrackModelIO() != 0)
    {
        ACLLITE_LOG_ERROR("Nanotrack model IO initialization failed");
        return -1;
    }

    model_initialized_ = true;
    ACLLITE_LOG_INFO("Nanotrack OM model initialized successfully");
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
                            tracking_validation_error_count_ = 0;
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

                bool needRedetection = false; // 是否触发重新检测
                if (tracking_validation_enabled_ && detectDataMsg->trackingActive)
                {
                    const DetectionOBB *bestPtr = nullptr; // 最佳检测框
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
                        tracking_validation_error_count_++;
                        ACLLITE_LOG_WARNING(
                            "[Tracking Ch%d Frame%d] Validation missing detection, error_count=%d",
                            detectDataMsg->channelId,
                            detectDataMsg->msgNum,
                            tracking_validation_error_count_);
                    }
                    else
                    {
                        float iou = ComputeIou(detectDataMsg->trackingResult.bbox, *bestPtr); // IOU值
                        if (iou < tracking_validation_iou_threshold_)
                        {
                            tracking_validation_error_count_++;
                            ACLLITE_LOG_WARNING(
                                "[Tracking Ch%d Frame%d] Validation IOU=%.3f below %.3f, error_count=%d",
                                detectDataMsg->channelId,
                                detectDataMsg->msgNum,
                                iou,
                                tracking_validation_iou_threshold_,
                                tracking_validation_error_count_);
                        }
                        else
                        {
                            tracking_validation_error_count_ = 0;
                        }
                    }

                    if (tracking_validation_error_count_ >= tracking_validation_max_errors_)
                    {
                        needRedetection = true;
                        tracking_validation_error_count_ = 0;
                        tracking_initialized_ = false;
                        track_loss_count_ = 0;
                        static_frame_count_ = 0;
                        has_last_box_ = false;
                        ACLLITE_LOG_INFO(
                            "[Tracking Ch%d] Validation failed %d times, requesting redetection",
                            detectDataMsg->channelId,
                            tracking_validation_max_errors_);
                    }
                }

                if (needRedetection)
                {
                    detectDataMsg->trackingActive = false;
                    detectDataMsg->needRedetection = true;
                    detectDataMsg->trackingResult.isTracked = false;
                    SendTrackingStateFeedback(detectDataMsg);
                }
                
                // Keep old fields for backward compatibility
                detectDataMsg->hasTracking = !needRedetection;
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
            tracking_validation_error_count_ = 0;
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
                tracking_validation_error_count_ = 0;
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
        this->cfg_.exemplar_size = size;
    }
}

void Tracking::setSearchSize(int size)
{
    if (size > 0)
    {
        this->cfg_.instance_size = size;
    }
}

void Tracking::setTemplateFactor(float factor)
{
    if (factor > 0.0f)
    {
        this->cfg_.context_amount = factor;
    }
}

void Tracking::setSearchFactor(float factor)
{
    if (factor > 0.0f)
    {
        this->search_scale_factor_ = factor;
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

void Tracking::setTrackingValidationEnabled(bool enabled)
{
    this->tracking_validation_enabled_ = enabled;
}

void Tracking::setTrackingValidationIouThreshold(float threshold)
{
    if (threshold >= 0.0f)
    {
        this->tracking_validation_iou_threshold_ = threshold;
    }
}

void Tracking::setTrackingValidationMaxErrors(int maxErrors)
{
    if (maxErrors > 0)
    {
        this->tracking_validation_max_errors_ = maxErrors;
    }
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

int Tracking::init(const cv::Mat &img, DrOBB bbox)
{
    if (!model_initialized_)
    {
        ACLLITE_LOG_ERROR("Model not initialized, call InitModel() first");
        return -1;
    }
    if (img.empty())
    {
        ACLLITE_LOG_ERROR("Init image is empty");
        return -1;
    }

    bbox.box.w = bbox.box.x1 - bbox.box.x0;
    bbox.box.h = bbox.box.y1 - bbox.box.y0;
    bbox.box.cx = bbox.box.x0 + 0.5f * bbox.box.w;
    bbox.box.cy = bbox.box.y0 + 0.5f * bbox.box.h;

    this->center_pos_ = cv::Point2f(
        bbox.box.x0 + (bbox.box.w - 1.f) * 0.5f,
        bbox.box.y0 + (bbox.box.h - 1.f) * 0.5f);
    this->size_ = cv::Point2f(bbox.box.w, bbox.box.h);

    float w_z =
        this->size_.x + this->cfg_.context_amount *
        (this->size_.x + this->size_.y);
    float h_z =
        this->size_.y + this->cfg_.context_amount *
        (this->size_.x + this->size_.y);
    float s_z = std::sqrt(w_z * h_z);
    this->channel_average_ = cv::mean(img);

    if (this->template_input_hw_.first > 0 &&
        this->template_input_hw_.first != this->cfg_.exemplar_size)
    {
        ACLLITE_LOG_ERROR("Nanotrack template size mismatch with backbone input");
        return -1;
    }

    auto z = GetSubwindow(img, this->center_pos_,
                          this->cfg_.exemplar_size,
                          static_cast<int>(std::round(s_z)),
                          this->channel_average_);
    this->zf_ = RunBackbone(z, this->zf_shape_);
    this->zf_ = AlignFeature(this->zf_, this->zf_shape_,
                             this->head_template_hw_, this->zf_shape_);
    if (this->zf_.empty())
    {
        ACLLITE_LOG_ERROR("Nanotrack backbone output is empty");
        return -1;
    }

    this->last_score_ = bbox.score > 0 ? bbox.score : 1.0f;
    this->state = bbox.box;
    this->object_box.box = bbox.box;
    this->object_box.score = this->last_score_;
    this->object_box.initScore =
        bbox.initScore > 0 ? bbox.initScore : bbox.score;
    this->object_box.class_id = bbox.class_id;
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
    if (img.empty() || this->zf_.empty())
    {
        ACLLITE_LOG_WARNING("Tracking input empty");
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    if (this->frame_id >= std::numeric_limits<int>::max())
    {
        this->frame_id = 0;
    }
    else
    {
        ++this->frame_id;
    }

    float w_z =
        this->size_.x + this->cfg_.context_amount *
        (this->size_.x + this->size_.y);
    float h_z =
        this->size_.y + this->cfg_.context_amount *
        (this->size_.x + this->size_.y);
    float s_z = std::sqrt(w_z * h_z);
    float scale_z = this->cfg_.exemplar_size / s_z;
    float s_x = s_z *
                (static_cast<float>(this->cfg_.instance_size) /
                 this->cfg_.exemplar_size) *
                this->search_scale_factor_;

    if (this->search_input_hw_.first > 0 &&
        this->search_input_hw_.first != this->cfg_.instance_size)
    {
        ACLLITE_LOG_ERROR("Nanotrack search size mismatch with backbone input");
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    auto x = GetSubwindow(img, this->center_pos_,
                          this->cfg_.instance_size,
                          static_cast<int>(std::round(s_x)),
                          this->channel_average_);

    std::vector<int64_t> xf_shape;
    std::vector<int64_t> cls_shape;
    std::vector<int64_t> loc_shape;
    auto xf = RunSearchBackbone(x, xf_shape);
    if (xf.empty())
    {
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }
    xf = AlignFeature(xf, xf_shape, this->head_search_hw_, xf_shape);
    auto head_outputs = RunHead(this->zf_, this->zf_shape_,
                                xf, xf_shape, cls_shape, loc_shape);
    auto cls = std::move(head_outputs.first);
    auto loc = std::move(head_outputs.second);

    if (cls_shape.size() >= 4)
    {
        EnsureScoreSize(static_cast<int>(cls_shape[2]));
    }

    auto score = ConvertScore(cls, cls_shape);
    auto pred_bbox = ConvertBBox(loc, loc_shape);
    if (score.empty() || pred_bbox.empty())
    {
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    auto change = [](float r) { return std::max(r, 1.f / r); };
    auto sz = [](float w, float h) {
        float pad = (w + h) * 0.5f;
        return std::sqrt((w + pad) * (h + pad));
    };

    std::vector<float> penalty(score.size());
    for (size_t i = 0; i < score.size(); ++i)
    {
        float sc =
            sz(pred_bbox[2 * score.size() + i],
               pred_bbox[3 * score.size() + i]) /
            sz(this->size_.x * scale_z, this->size_.y * scale_z);
        float rc = (this->size_.x / this->size_.y) /
                   (pred_bbox[2 * score.size() + i] /
                    pred_bbox[3 * score.size() + i]);
        penalty[i] = std::exp(-(change(sc) * change(rc) - 1.f) *
                              this->cfg_.penalty_k);
    }

    std::vector<float> pscore(score.size());
    for (size_t i = 0; i < score.size(); ++i)
    {
        float s = penalty[i] * score[i];
        pscore[i] = s * (1.f - this->cfg_.window_influence) +
                    this->window_[i] * this->cfg_.window_influence;
    }

    auto best_iter = std::max_element(pscore.begin(), pscore.end());
    size_t best_idx =
        static_cast<size_t>(std::distance(pscore.begin(), best_iter));

    cv::Point2f bbox;
    bbox.x = pred_bbox[best_idx] / scale_z + this->center_pos_.x;
    bbox.y = pred_bbox[score.size() + best_idx] / scale_z + this->center_pos_.y;

    float width =
        this->size_.x * (1 - this->cfg_.lr) +
        pred_bbox[2 * score.size() + best_idx] / scale_z * this->cfg_.lr;
    float height =
        this->size_.y * (1 - this->cfg_.lr) +
        pred_bbox[3 * score.size() + best_idx] / scale_z * this->cfg_.lr;

    auto clipped =
        BboxClip(bbox.x, bbox.y, width, height, img.rows, img.cols);
    this->center_pos_ = cv::Point2f(clipped[0], clipped[1]);
    this->size_ = cv::Point2f(clipped[2], clipped[3]);

    DrBBox out_box;
    out_box.cx = this->center_pos_.x;
    out_box.cy = this->center_pos_.y;
    out_box.w = this->size_.x;
    out_box.h = this->size_.y;
    out_box.x0 = out_box.cx - 0.5f * out_box.w;
    out_box.y0 = out_box.cy - 0.5f * out_box.h;
    out_box.x1 = out_box.x0 + out_box.w;
    out_box.y1 = out_box.y0 + out_box.h;

    float init_score = this->object_box.initScore;
    int class_id = this->object_box.class_id;
    this->last_score_ = score[best_idx];
    this->state = out_box;
    this->object_box.box = out_box;
    this->object_box.score = this->last_score_;
    this->object_box.initScore = init_score;
    this->object_box.class_id = class_id;

    return this->object_box;
}

void Tracking::InitNanotrackModelPath(const std::string &model_path)
{
    if (model_path.empty())
    {
        head_model_path_ = kDefaultNanotrackHeadModel;
        backbone_model_path_ = kDefaultNanotrackBackboneModel;
        search_model_path_ = kDefaultNanotrackBackboneSearchModel;
        return;
    }

    std::vector<std::string> parts;
    char delimiter = '\0';
    if (model_path.find(';') != std::string::npos)
    {
        delimiter = ';';
    }
    else if (model_path.find(',') != std::string::npos)
    {
        delimiter = ',';
    }

    if (delimiter != '\0')
    {
        std::stringstream ss(model_path);
        std::string item;
        while (std::getline(ss, item, delimiter))
        {
            if (!item.empty())
            {
                parts.push_back(item);
            }
        }
    }

    if (parts.size() >= 3)
    {
        head_model_path_ = parts[0];
        backbone_model_path_ = parts[1];
        search_model_path_ = parts[2];
        return;
    }

    head_model_path_ = model_path;
    std::string dir;
    size_t pos = model_path.find_last_of("/\\");
    if (pos != std::string::npos)
    {
        dir = model_path.substr(0, pos + 1);
    }
    backbone_model_path_ = dir + "nanotrack_backbone_bs1.om";
    search_model_path_ = dir + "nanotrack_backbone_search_bs1.om";
}

int Tracking::InitNanotrackModelIO()
{
    backbone_input_size_ =
        backbone_model_.GetModelInputSize(0) / sizeof(float);
    template_input_hw_ = CalcSquareHW(backbone_input_size_, 3);
    if (template_input_hw_.first > 0)
    {
        cfg_.exemplar_size = template_input_hw_.first;
    }

    std::vector<ModelOutputInfo> backbone_outputs;
    if (backbone_model_.GetModelOutputInfo(backbone_outputs) != ACLLITE_OK ||
        backbone_outputs.empty())
    {
        ACLLITE_LOG_ERROR("Backbone output info not available");
        return -1;
    }
    backbone_output_shape_ = DimsToShape(backbone_outputs[0].dims);
    backbone_output_size_ = 1;
    for (size_t i = 0; i < backbone_output_shape_.size(); ++i)
    {
        if (backbone_output_shape_[i] <= 0)
        {
            backbone_output_size_ = 0;
            break;
        }
        backbone_output_size_ *= static_cast<size_t>(backbone_output_shape_[i]);
    }
    backbone_output_.resize(backbone_output_size_);

    if (has_search_backbone_)
    {
        search_input_size_ =
            search_model_.GetModelInputSize(0) / sizeof(float);
        search_input_hw_ = CalcSquareHW(search_input_size_, 3);

        std::vector<ModelOutputInfo> search_outputs;
        if (search_model_.GetModelOutputInfo(search_outputs) != ACLLITE_OK ||
            search_outputs.empty())
        {
            ACLLITE_LOG_ERROR("Search backbone output info not available");
            return -1;
        }
        search_output_shape_ = DimsToShape(search_outputs[0].dims);
        search_output_size_ = 1;
        for (size_t i = 0; i < search_output_shape_.size(); ++i)
        {
            if (search_output_shape_[i] <= 0)
            {
                search_output_size_ = 0;
                break;
            }
            search_output_size_ *=
                static_cast<size_t>(search_output_shape_[i]);
        }
        search_output_.resize(search_output_size_);
    }
    else
    {
        search_input_size_ = backbone_input_size_;
        search_input_hw_ = template_input_hw_;
        search_output_shape_ = backbone_output_shape_;
        search_output_size_ = backbone_output_size_;
        search_output_.resize(search_output_size_);
    }

    if (search_input_hw_.first > 0)
    {
        cfg_.instance_size = search_input_hw_.first;
    }

    head_input_z_size_ =
        head_model_.GetModelInputSize(0) / sizeof(float);
    head_input_x_size_ =
        head_model_.GetModelInputSize(1) / sizeof(float);

    auto input0_hw = CalcSquareHW(head_input_z_size_, 96);
    auto input1_hw = CalcSquareHW(head_input_x_size_, 96);
    if (input0_hw.first > 0 && input1_hw.first > 0)
    {
        if (input0_hw.first <= input1_hw.first)
        {
            head_template_hw_ = input0_hw;
            head_search_hw_ = input1_hw;
            head_input_z_index_ = 0;
            head_input_x_index_ = 1;
        }
        else
        {
            head_template_hw_ = input1_hw;
            head_search_hw_ = input0_hw;
            head_input_z_index_ = 1;
            head_input_x_index_ = 0;
            std::swap(head_input_z_size_, head_input_x_size_);
        }
    }

    std::vector<ModelOutputInfo> head_outputs;
    if (head_model_.GetModelOutputInfo(head_outputs) != ACLLITE_OK ||
        head_outputs.size() < 2)
    {
        ACLLITE_LOG_ERROR("Head output info not available");
        return -1;
    }

    head_output_cls_index_ = 0;
    head_output_loc_index_ = 1;
    for (size_t i = 0; i < head_outputs.size(); ++i)
    {
        auto shape = DimsToShape(head_outputs[i].dims);
        if (shape.size() >= 2 && shape[1] == 4)
        {
            head_output_loc_index_ = static_cast<int>(i);
            head_loc_shape_ = shape;
        }
        else
        {
            head_output_cls_index_ = static_cast<int>(i);
            head_cls_shape_ = shape;
        }
    }

    if (head_cls_shape_.empty())
    {
        head_cls_shape_ = DimsToShape(head_outputs[head_output_cls_index_].dims);
    }
    if (head_loc_shape_.empty())
    {
        head_loc_shape_ = DimsToShape(head_outputs[head_output_loc_index_].dims);
    }

    head_output_cls_size_ = 1;
    for (size_t i = 0; i < head_cls_shape_.size(); ++i)
    {
        if (head_cls_shape_[i] <= 0)
        {
            head_output_cls_size_ = 0;
            break;
        }
        head_output_cls_size_ *= static_cast<size_t>(head_cls_shape_[i]);
    }

    head_output_loc_size_ = 1;
    for (size_t i = 0; i < head_loc_shape_.size(); ++i)
    {
        if (head_loc_shape_[i] <= 0)
        {
            head_output_loc_size_ = 0;
            break;
        }
        head_output_loc_size_ *= static_cast<size_t>(head_loc_shape_[i]);
    }

    head_output_cls_.resize(head_output_cls_size_);
    head_output_loc_.resize(head_output_loc_size_);

    if (head_cls_shape_.size() >= 4)
    {
        EnsureScoreSize(static_cast<int>(head_cls_shape_[2]));
    }

    return 0;
}

std::vector<float> Tracking::RunBackbone(const std::vector<float> &input,
                                         std::vector<int64_t> &out_shape)
{
    if (input.size() != backbone_input_size_)
    {
        out_shape.clear();
        return {};
    }

    std::vector<DataInfo> inputData;
    DataInfo template_input;
    template_input.data = const_cast<float *>(input.data());
    template_input.size = input.size() * sizeof(float);
    inputData.push_back(template_input);

    AclLiteError ret = backbone_model_.CreateInput(inputData);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Create backbone input failed, error: %d", ret);
        out_shape.clear();
        return {};
    }

    std::vector<InferenceOutput> outputs;
    ret = backbone_model_.ExecuteV2(outputs);
    if (ret != ACLLITE_OK || outputs.empty())
    {
        ACLLITE_LOG_ERROR("Execute backbone failed, error: %d", ret);
        backbone_model_.DestroyInput();
        out_shape.clear();
        return {};
    }

    if (outputs[0].size >= backbone_output_size_ * sizeof(float))
    {
        void *hostBuffer = CopyDataToHost(
            outputs[0].data.get(),
            backbone_output_size_ * sizeof(float),
            runMode_,
            MEMORY_NORMAL);
        if (hostBuffer != nullptr)
        {
            std::memcpy(backbone_output_.data(),
                        hostBuffer,
                        backbone_output_size_ * sizeof(float));
            delete[] static_cast<uint8_t *>(hostBuffer);
        }
        else
        {
            ACLLITE_LOG_ERROR("Copy backbone output to host failed");
            backbone_output_.clear();
        }
    }
    else
    {
        ACLLITE_LOG_ERROR("Backbone output size mismatch: expected %zu, got %u",
                          backbone_output_size_ * sizeof(float),
                          outputs[0].size);
        backbone_output_.clear();
    }

    backbone_model_.DestroyInput();
    out_shape = backbone_output_shape_;
    return backbone_output_;
}

std::vector<float> Tracking::RunSearchBackbone(
    const std::vector<float> &input,
    std::vector<int64_t> &out_shape)
{
    if (input.size() != search_input_size_)
    {
        out_shape.clear();
        return {};
    }

    AclLiteModel &model = has_search_backbone_ ? search_model_ : backbone_model_;
    std::vector<float> &output =
        has_search_backbone_ ? search_output_ : backbone_output_;
    size_t output_size =
        has_search_backbone_ ? search_output_size_ : backbone_output_size_;
    const std::vector<int64_t> &shape =
        has_search_backbone_ ? search_output_shape_ : backbone_output_shape_;

    std::vector<DataInfo> inputData;
    DataInfo search_input;
    search_input.data = const_cast<float *>(input.data());
    search_input.size = input.size() * sizeof(float);
    inputData.push_back(search_input);

    AclLiteError ret = model.CreateInput(inputData);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Create search backbone input failed, error: %d", ret);
        out_shape.clear();
        return {};
    }

    std::vector<InferenceOutput> outputs;
    ret = model.ExecuteV2(outputs);
    if (ret != ACLLITE_OK || outputs.empty())
    {
        ACLLITE_LOG_ERROR("Execute search backbone failed, error: %d", ret);
        model.DestroyInput();
        out_shape.clear();
        return {};
    }

    if (outputs[0].size >= output_size * sizeof(float))
    {
        void *hostBuffer = CopyDataToHost(
            outputs[0].data.get(),
            output_size * sizeof(float),
            runMode_,
            MEMORY_NORMAL);
        if (hostBuffer != nullptr)
        {
            std::memcpy(output.data(),
                        hostBuffer,
                        output_size * sizeof(float));
            delete[] static_cast<uint8_t *>(hostBuffer);
        }
        else
        {
            ACLLITE_LOG_ERROR("Copy search backbone output to host failed");
            output.clear();
        }
    }
    else
    {
        ACLLITE_LOG_ERROR("Search backbone output size mismatch: expected %zu, got %u",
                          output_size * sizeof(float),
                          outputs[0].size);
        output.clear();
    }

    model.DestroyInput();
    out_shape = shape;
    return output;
}

std::pair<std::vector<float>, std::vector<float>>
Tracking::RunHead(const std::vector<float> &zf,
                  const std::vector<int64_t> &zf_shape,
                  const std::vector<float> &xf,
                  const std::vector<int64_t> &xf_shape,
                  std::vector<int64_t> &cls_shape,
                  std::vector<int64_t> &loc_shape)
{
    (void)zf_shape;
    (void)xf_shape;

    if (zf.size() != head_input_z_size_ || xf.size() != head_input_x_size_)
    {
        cls_shape.clear();
        loc_shape.clear();
        return {};
    }

    std::vector<DataInfo> inputData(2);
    if (head_input_z_index_ == 0)
    {
        inputData[0] = {const_cast<float *>(zf.data()),
                        static_cast<uint32_t>(zf.size() * sizeof(float))};
        inputData[1] = {const_cast<float *>(xf.data()),
                        static_cast<uint32_t>(xf.size() * sizeof(float))};
    }
    else
    {
        inputData[0] = {const_cast<float *>(xf.data()),
                        static_cast<uint32_t>(xf.size() * sizeof(float))};
        inputData[1] = {const_cast<float *>(zf.data()),
                        static_cast<uint32_t>(zf.size() * sizeof(float))};
    }

    AclLiteError ret = head_model_.CreateInput(inputData);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Create head input failed, error: %d", ret);
        cls_shape.clear();
        loc_shape.clear();
        return {};
    }

    std::vector<InferenceOutput> outputs;
    ret = head_model_.ExecuteV2(outputs);
    if (ret != ACLLITE_OK || outputs.size() < 2)
    {
        ACLLITE_LOG_ERROR("Execute head failed, error: %d", ret);
        head_model_.DestroyInput();
        cls_shape.clear();
        loc_shape.clear();
        return {};
    }

    if (outputs[head_output_cls_index_].size >=
        head_output_cls_size_ * sizeof(float))
    {
        void *hostBuffer = CopyDataToHost(
            outputs[head_output_cls_index_].data.get(),
            head_output_cls_size_ * sizeof(float),
            runMode_,
            MEMORY_NORMAL);
        if (hostBuffer != nullptr)
        {
            std::memcpy(head_output_cls_.data(),
                        hostBuffer,
                        head_output_cls_size_ * sizeof(float));
            delete[] static_cast<uint8_t *>(hostBuffer);
        }
        else
        {
            ACLLITE_LOG_ERROR("Copy head cls output to host failed");
        }
    }
    else
    {
        ACLLITE_LOG_ERROR("Head cls output size mismatch: expected %zu, got %u",
                          head_output_cls_size_ * sizeof(float),
                          outputs[head_output_cls_index_].size);
        head_output_cls_.clear();
    }

    if (outputs[head_output_loc_index_].size >=
        head_output_loc_size_ * sizeof(float))
    {
        void *hostBuffer = CopyDataToHost(
            outputs[head_output_loc_index_].data.get(),
            head_output_loc_size_ * sizeof(float),
            runMode_,
            MEMORY_NORMAL);
        if (hostBuffer != nullptr)
        {
            std::memcpy(head_output_loc_.data(),
                        hostBuffer,
                        head_output_loc_size_ * sizeof(float));
            delete[] static_cast<uint8_t *>(hostBuffer);
        }
        else
        {
            ACLLITE_LOG_ERROR("Copy head loc output to host failed");
        }
    }
    else
    {
        ACLLITE_LOG_ERROR("Head loc output size mismatch: expected %zu, got %u",
                          head_output_loc_size_ * sizeof(float),
                          outputs[head_output_loc_index_].size);
        head_output_loc_.clear();
    }

    head_model_.DestroyInput();
    cls_shape = head_cls_shape_;
    loc_shape = head_loc_shape_;
    return {head_output_cls_, head_output_loc_};
}

void Tracking::EnsureScoreSize(int size)
{
    if (size > 0 && size != this->cfg_.score_size)
    {
        this->cfg_.score_size = size;
    }

    if (this->cfg_.score_size > 0)
    {
        this->window_ = BuildWindow(this->cfg_.score_size);
        this->points_ = BuildPoints(this->cfg_.stride, this->cfg_.score_size);
    }
}

std::vector<float> Tracking::BuildWindow(int size)
{
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<float> hanning(size);
    for (int i = 0; i < size; ++i)
    {
        hanning[i] = 0.5f - 0.5f * std::cos(2.f * kPi * i / (size - 1));
    }
    std::vector<float> window(size * size);
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            window[y * size + x] = hanning[y] * hanning[x];
        }
    }
    return window;
}

std::vector<cv::Point2f> Tracking::BuildPoints(int stride, int size)
{
    std::vector<cv::Point2f> pts;
    pts.reserve(static_cast<size_t>(size * size));
    int ori = -(size / 2) * stride;
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            pts.emplace_back(static_cast<float>(ori + stride * x),
                             static_cast<float>(ori + stride * y));
        }
    }
    return pts;
}

std::vector<float> Tracking::GetSubwindow(const cv::Mat &img,
                                          const cv::Point2f &pos,
                                          int model_sz,
                                          int original_sz,
                                          const cv::Scalar &avg_chans)
{
    float c = (original_sz + 1) * 0.5f;
    int context_xmin = static_cast<int>(std::floor(pos.x - c + 0.5f));
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = static_cast<int>(std::floor(pos.y - c + 0.5f));
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = std::max(0, -context_xmin);
    int top_pad = std::max(0, -context_ymin);
    int right_pad = std::max(0, context_xmax - img.cols + 1);
    int bottom_pad = std::max(0, context_ymax - img.rows + 1);

    cv::Mat te_im;
    if (left_pad || top_pad || right_pad || bottom_pad)
    {
        te_im = cv::Mat(img.rows + top_pad + bottom_pad,
                        img.cols + left_pad + right_pad, img.type(), avg_chans);
        img.copyTo(te_im(cv::Rect(left_pad, top_pad, img.cols, img.rows)));
    }
    else
    {
        te_im = img;
    }

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Rect roi(context_xmin, context_ymin,
                 context_xmax - context_xmin + 1,
                 context_ymax - context_ymin + 1);
    cv::Mat im_patch = te_im(roi).clone();
    if (model_sz != original_sz)
    {
        cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));
    }

    std::vector<float> data(1 * 3 * model_sz * model_sz);
    for (int cidx = 0; cidx < 3; ++cidx)
    {
        for (int y = 0; y < model_sz; ++y)
        {
            const uint8_t *row_ptr = im_patch.ptr<uint8_t>(y);
            for (int x = 0; x < model_sz; ++x)
            {
                data[cidx * model_sz * model_sz + y * model_sz + x] =
                    static_cast<float>(row_ptr[x * 3 + cidx]);
            }
        }
    }

    this->subwindow_shape_ = {1, 3, model_sz, model_sz};
    return data;
}

std::vector<float> Tracking::AlignFeature(
    const std::vector<float> &feat, const std::vector<int64_t> &shape,
    const std::pair<int, int> &target_hw, std::vector<int64_t> &out_shape)
{
    if (target_hw.first <= 0 || target_hw.second <= 0 ||
        shape.size() < 4)
    {
        out_shape = shape;
        return feat;
    }

    int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];
    int t_h = target_hw.first;
    int t_w = target_hw.second;
    if (h == t_h && w == t_w)
    {
        out_shape = shape;
        return feat;
    }
    if (h < t_h || w < t_w)
    {
        out_shape = shape;
        return feat;
    }

    int h_start = static_cast<int>((h - t_h) / 2);
    int w_start = static_cast<int>((w - t_w) / 2);
    std::vector<float> cropped(static_cast<size_t>(n * c * t_h * t_w));
    for (int64_t nc = 0; nc < n * c; ++nc)
    {
        int64_t base_in = nc * h * w;
        int64_t base_out = nc * t_h * t_w;
        for (int yy = 0; yy < t_h; ++yy)
        {
            for (int xx = 0; xx < t_w; ++xx)
            {
                cropped[base_out + yy * t_w + xx] =
                    feat[base_in + (yy + h_start) * w + (xx + w_start)];
            }
        }
    }
    out_shape = {n, c, t_h, t_w};
    return cropped;
}

std::vector<float> Tracking::ConvertScore(
    const std::vector<float> &cls,
    const std::vector<int64_t> &shape) const
{
    if (shape.size() < 4)
    {
        return {};
    }

    int64_t c = shape[1];
    int64_t h = shape[2];
    int64_t w = shape[3];
    int64_t hw = h * w;
    std::vector<float> score(static_cast<size_t>(hw), 0.f);
    if (c == 1)
    {
        for (int64_t i = 0; i < hw; ++i)
        {
            score[i] = 1.f / (1.f + std::exp(-cls[i]));
        }
    }
    else
    {
        for (int64_t y = 0; y < h; ++y)
        {
            for (int64_t x = 0; x < w; ++x)
            {
                int64_t idx = y * w + x;
                float s0 = cls[idx];
                float s1 = cls[hw + idx];
                float e0 = std::exp(s0);
                float e1 = std::exp(s1);
                score[idx] = e1 / (e0 + e1 + 1e-6f);
            }
        }
    }
    return score;
}

std::vector<float> Tracking::ConvertBBox(
    const std::vector<float> &loc,
    const std::vector<int64_t> &shape) const
{
    if (shape.size() < 4 || this->points_.empty())
    {
        return {};
    }

    int64_t h = shape[2];
    int64_t w = shape[3];
    int64_t hw = h * w;
    if (static_cast<size_t>(hw) != this->points_.size())
    {
        return {};
    }

    std::vector<float> bbox(static_cast<size_t>(4 * hw));
    for (int64_t y = 0; y < h; ++y)
    {
        for (int64_t x = 0; x < w; ++x)
        {
            int64_t idx = y * w + x;
            float l = loc[idx];
            float t = loc[hw + idx];
            float r = loc[2 * hw + idx];
            float b = loc[3 * hw + idx];
            float x1 = this->points_[idx].x - l;
            float y1 = this->points_[idx].y - t;
            float x2 = this->points_[idx].x + r;
            float y2 = this->points_[idx].y + b;
            bbox[idx] = (x1 + x2) * 0.5f;
            bbox[hw + idx] = (y1 + y2) * 0.5f;
            bbox[2 * hw + idx] = x2 - x1;
            bbox[3 * hw + idx] = y2 - y1;
        }
    }
    return bbox;
}

std::array<float, 4> Tracking::BboxClip(float cx, float cy, float width,
                                        float height, int rows, int cols) const
{
    cx = std::max(0.f, std::min(cx, static_cast<float>(cols)));
    cy = std::max(0.f, std::min(cy, static_cast<float>(rows)));
    width = std::max(10.f, std::min(width, static_cast<float>(cols)));
    height = std::max(10.f, std::min(height, static_cast<float>(rows)));
    return {cx, cy, width, height};
}

std::vector<int64_t> Tracking::DimsToShape(const aclmdlIODims &dims) const
{
    std::vector<int64_t> shape;
    for (int i = 0; i < dims.dimCount; ++i)
    {
        shape.push_back(dims.dims[i]);
    }
    return shape;
}

std::pair<int, int> Tracking::CalcSquareHW(size_t elements,
                                           int channels) const
{
    if (channels <= 0)
    {
        return {-1, -1};
    }
    size_t hw = elements / static_cast<size_t>(channels);
    int side = static_cast<int>(std::sqrt(static_cast<float>(hw)));
    if (side <= 0 || static_cast<size_t>(side * side) != hw)
    {
        return {-1, -1};
    }
    return {side, side};
}

float Tracking::ComputeIou(const DetectionOBB &a, const DetectionOBB &b) const
{
    float inter_x0 = std::max(a.x0, b.x0); // 交集左上角x
    float inter_y0 = std::max(a.y0, b.y0); // 交集左上角y
    float inter_x1 = std::min(a.x1, b.x1); // 交集右下角x
    float inter_y1 = std::min(a.y1, b.y1); // 交集右下角y

    float inter_w = std::max(0.0f, inter_x1 - inter_x0); // 交集宽
    float inter_h = std::max(0.0f, inter_y1 - inter_y0); // 交集高
    float inter_area = inter_w * inter_h; // 交集面积

    float area_a = std::max(0.0f, a.x1 - a.x0) * std::max(0.0f, a.y1 - a.y0); // 框A面积
    float area_b = std::max(0.0f, b.x1 - b.x0) * std::max(0.0f, b.y1 - b.y0); // 框B面积
    float union_area = area_a + area_b - inter_area; // 并集面积
    if (union_area <= 0.0f)
    {
        return 0.0f;
    }
    return inter_area / union_area;
}
