#include "mixformerv2_om.h"
#include "AclLiteApp.h"
#include "AclLiteUtils.h"
#include "Params.h"
#include <cstring>
#include <limits>
#include <vector>

// Constants for normalization
const float MixformerV2OM::mean_vals[3] = {
    0.485f * 255,
    0.456f * 255,
    0.406f * 255};
const float MixformerV2OM::norm_vals[3] = {
    1.0f / 0.229f,
    1.0f / 0.224f,
    1.0f / 0.225f};

namespace
{
constexpr const char *kDefaultMixFormerModel =
    "model/mixformerv2_online_small.om";

// Model input/output tensor names (these should match your OM model)
constexpr const char *kInputTemplateName = "template";
constexpr const char *kInputOnlineTemplateName = "online_template";
constexpr const char *kInputSearchName = "search";
constexpr const char *kOutputBoxesName = "pred_boxes";
constexpr const char *kOutputScoresName = "pred_scores";
} // namespace

MixformerV2OM::MixformerV2OM(const std::string &model_path)
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

MixformerV2OM::~MixformerV2OM()
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

int MixformerV2OM::InitModel()
{
    if (model_initialized_)
    {
        ACLLITE_LOG_WARNING("Model already initialized");
        return 0;
    }

    AclLiteError ret = model_.Init(model_path_);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("MixFormerV2 OM model init failed, error: %d", ret);
        return -1;
    }

    model_initialized_ = true;
    ACLLITE_LOG_INFO("MixFormerV2 OM model initialized successfully");
    return 0;
}

// AclLiteThread lifecycle
AclLiteError MixformerV2OM::Init()
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

AclLiteError MixformerV2OM::Process(int msgId, std::shared_ptr<void> data)
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

        // 单目标跟踪：首次检测初始化，后续调用 track 更新
        if (!detectDataMsg->frame.empty())
        {
            cv::Mat &img = detectDataMsg->frame[0];

            if (!tracking_initialized_)
            {
                // 选择最高分目标作为跟踪目标
                if (!detectDataMsg->detections.empty())
                {
                    const auto best = *std::max_element(
                        detectDataMsg->detections.begin(),
                        detectDataMsg->detections.end(),
                        [](const DetectionOBB &a, const DetectionOBB &b)
                        { return a.score < b.score; });

                    DrOBB initBox;
                    initBox.box.x0 = best.x0;
                    initBox.box.y0 = best.y0;
                    initBox.box.x1 = best.x1;
                    initBox.box.y1 = best.y1;
                    initBox.class_id = best.class_id;

                    if (this->init(img, initBox) == 0)
                    {
                        tracking_initialized_ = true;
                        detectDataMsg->hasTracking = true;
                        detectDataMsg->trackInitScore = best.score;
                    }
                }
            }
            else
            {
                const DrOBB &tracked = this->track(img);
                // 将跟踪结果追加到 detections，供输出线程使用
                DetectionOBB detOut;
                detOut.x0 = tracked.box.x0;
                detOut.y0 = tracked.box.y0;
                detOut.x1 = tracked.box.x1;
                detOut.y1 = tracked.box.y1;
                detOut.score = tracked.score;
                detOut.class_id = tracked.class_id;
                detectDataMsg->detections.push_back(detOut);
                detectDataMsg->hasTracking = true;
                detectDataMsg->trackScore = tracked.score;
            }
        }

        // Forward message to DataOutputThread
        AclLiteError ret;
        while (1)
        {
            ret = SendMessage(
                dataOutputThreadId_,
                MSG_OUTPUT_FRAME,
                detectDataMsg);
            if (ret == ACLLITE_ERROR_ENQUEUE)
            {
                usleep(500);
                continue;
            }
            break;
        }

        // Handle last frame encoding finish
        if (detectDataMsg->isLastFrame)
        {
            while (1)
            {
                ret = SendMessage(
                    dataOutputThreadId_,
                    MSG_ENCODE_FINISH,
                    detectDataMsg);
                if (ret == ACLLITE_ERROR_ENQUEUE)
                {
                    usleep(500);
                    continue;
                }
                break;
            }
        }

        return ACLLITE_OK;
    }
    default:
        ACLLITE_LOG_INFO("MixformerV2OM thread ignore msg %d", msgId);
        return ACLLITE_OK;
    }
}

void MixformerV2OM::setTemplateSize(int size)
{
    if (size > 0)
    {
        this->template_size = size;
    }
}

void MixformerV2OM::setSearchSize(int size)
{
    if (size > 0)
    {
        this->search_size = size;
    }
}

void MixformerV2OM::setTemplateFactor(float factor)
{
    if (factor > 0.0f)
    {
        this->template_factor = factor;
    }
}

void MixformerV2OM::setSearchFactor(float factor)
{
    if (factor > 0.0f)
    {
        this->search_factor = factor;
    }
}

void MixformerV2OM::setUpdateInterval(int interval)
{
    this->update_interval = interval;
}

void MixformerV2OM::setTemplateUpdateScoreThreshold(float threshold)
{
    this->template_update_score_threshold = threshold;
}

void MixformerV2OM::setMaxScoreDecay(float decay)
{
    this->max_score_decay = decay;
}

void MixformerV2OM::resetMaxPredScore() { this->max_pred_score = 0.0f; }

int MixformerV2OM::init(const cv::Mat &img, DrOBB bbox)
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
    this->object_box.score = 1.0f;
    this->object_box.class_id = bbox.class_id;
    this->resetMaxPredScore();
    this->frame_id = 0;

    return 0;
}

const DrOBB &MixformerV2OM::track(const cv::Mat &img)
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

void MixformerV2OM::infer()
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
    // ExecuteV2 returns data on device memory, need to copy to host
    // Assuming outputs are in order: pred_boxes, pred_scores
    if (inferenceOutput.size() >= 2)
    {
        // Get run mode for data copying
        aclrtRunMode runMode;
        aclError     aclRet = aclrtGetRunMode(&runMode);
        if (aclRet != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("Get run mode failed");
            model_.DestroyInput();
            return;
        }

        // Copy pred_boxes output from device to host
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
int MixformerV2OM::sample_target(
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

    int x = bbox.x0;
    int y = bbox.y0;
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

void MixformerV2OM::half_norm(const cv::Mat &img, float *output)
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

DrBBox MixformerV2OM::cal_bbox(
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

void MixformerV2OM::map_box_back(
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

void MixformerV2OM::clip_box(DrBBox &box, int img_h, int img_w, int border)
{
    box.x0 = std::min(std::max(0, int(box.x0)), img_w - border);
    box.y0 = std::min(std::max(0, int(box.y0)), img_h - border);
    box.x1 = std::min(std::max(border, int(box.x1)), img_w);
    box.y1 = std::min(std::max(border, int(box.y1)), img_h);
}

void MixformerV2OM::setInputTemplateData(const float *data, size_t size)
{
    if (data && size == input_template_size)
    {
        std::memcpy(input_template, data, size * sizeof(float));
    }
}

void MixformerV2OM::setInputOnlineTemplateData(const float *data, size_t size)
{
    if (data && size == input_online_template_size)
    {
        std::memcpy(input_online_template, data, size * sizeof(float));
    }
}

void MixformerV2OM::setInputSearchData(const float *data, size_t size)
{
    if (data && size == input_search_size)
    {
        std::memcpy(input_search, data, size * sizeof(float));
    }
}

std::vector<float> MixformerV2OM::getOutputPredBoxes() const
{
    std::vector<float> result(output_pred_boxes,
                               output_pred_boxes + output_pred_boxes_size);
    return result;
}

std::vector<float> MixformerV2OM::getOutputPredScores() const
{
    std::vector<float> result(output_pred_scores,
                               output_pred_scores + output_pred_scores_size);
    return result;
}
