#include "tracking/tracking.h"
#include "AclLiteResource.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <sstream>

// Lightweight .npy reader for float32 arrays. Supports v1.0 and v2.0 headers.
static uint16_t read_le_u16(std::ifstream &f) {
    uint8_t b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    return uint16_t(b[0]) | (uint16_t(b[1]) << 8);
}
static uint32_t read_le_u32(std::ifstream &f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}

// Safe manual integer parse from ASCII string without libc strtol/stoi usage.
static bool try_parse_int(const std::string &s, int &out) {
    if (s.empty()) return false;
    size_t i = 0;
    bool neg = false;
    if (s[0] == '+' || s[0] == '-') {
        neg = (s[0] == '-');
        i = 1;
        if (i >= s.size()) return false; // just a sign
    }
    long long val = 0;
    for (; i < s.size(); ++i) {
        char c = s[i];
        if (c < '0' || c > '9') return false;
        val = val * 10 + (c - '0');
        // Keep it in range of 64-bit to avoid overflows
        if (val > (1LL<<62)) return false;
    }
    if (neg) val = -val;
    out = static_cast<int>(val);
    return true;
}

std::vector<float> load_npy(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read magic string
    char magic[6];
    file.read(magic, 6);
    if (!file || std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Not a .npy file (bad magic): " + path);
    }

    // Read version
    unsigned char ver[2];
    file.read(reinterpret_cast<char*>(ver), 2);
    if (!file) {
        throw std::runtime_error("Failed to read .npy version: " + path);
    }
    int major = ver[0];
    int minor = ver[1];

    // Read header length (uint16 for v1.x, uint32 for v2.x+)
    size_t header_len = 0;
    if (major == 1) {
        header_len = static_cast<size_t>(read_le_u16(file));
    } else if (major == 2) {
        header_len = static_cast<size_t>(read_le_u32(file));
    } else {
        throw std::runtime_error("Unsupported .npy version: " + std::to_string(major) + "." + std::to_string(minor));
    }

    // Read header
    std::string header;
    header.resize(header_len);
    file.read(&header[0], header_len);
    if (!file) {
        throw std::runtime_error("Failed to read .npy header: " + path);
    }

    // Parse 'descr' (dtype) to ensure we have float32
    // header example: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 3, 112, 112), }"
    auto find_key = [&](const std::string &key)->size_t {
        // Accept 'key' or "key"
        std::string key1 = "'" + key + "'";
        std::string key2 = '"' + key + '"';
        size_t pos = header.find(key1);
        if (pos != std::string::npos) return pos;
        return header.find(key2);
    };

    // Parse dtype / descr
    size_t descr_pos = find_key("descr");
    if (descr_pos == std::string::npos) {
        throw std::runtime_error("Invalid header: no descr found in: " + header);
    }
    size_t colon = header.find(':', descr_pos);
    if (colon == std::string::npos) {
        throw std::runtime_error("Invalid header: malformed descr: " + header);
    }
    // find opening quote after colon
    size_t qstart = header.find_first_of("'\"", colon);
    size_t qend = std::string::npos;
    if (qstart != std::string::npos) qend = header.find_first_of("'\"", qstart+1);
    if (qstart == std::string::npos || qend == std::string::npos) {
        throw std::runtime_error("Invalid header: malformed descr quotes: " + header);
    }
    std::string descr = header.substr(qstart+1, qend - qstart - 1);
    // Expect '<f4' or '|f4' etc. Only accept float32
    if (descr.find("f4") == std::string::npos) {
        throw std::runtime_error("Unsupported dtype: " + descr);
    }

    // Parse shape
    size_t shape_pos = find_key("shape");
    if (shape_pos == std::string::npos) {
        throw std::runtime_error("Invalid header: no shape found in: " + header);
    }
    size_t paren = header.find('(', shape_pos);
    if (paren == std::string::npos) {
        throw std::runtime_error("Invalid header: shape not opened in: " + header);
    }
    size_t close = header.find(')', paren);
    if (close == std::string::npos) {
        throw std::runtime_error("Invalid header: shape not closed in: " + header);
    }
    std::string shape_str = header.substr(paren+1, close - paren - 1);

    // Extract numbers (dimensions) supporting commas and whitespace
    std::vector<size_t> dims;
    size_t idx = 0;
    while (idx < shape_str.size()) {
        // skip spaces
        while (idx < shape_str.size() && isspace((unsigned char)shape_str[idx])) ++idx;
        if (idx >= shape_str.size()) break;
        // read number
        size_t start = idx;
        while (idx < shape_str.size() && (shape_str[idx] == '+' || shape_str[idx] == '-' || (shape_str[idx] >= '0' && shape_str[idx] <= '9'))) ++idx;
        std::string num_str = shape_str.substr(start, idx - start);
        if (!num_str.empty()) {
            // Manual parse of integer to avoid libc strtol dependency
            if (num_str.empty()) {
                throw std::runtime_error(std::string("Invalid header: empty shape number in: ") + header);
            }
            bool neg = false;
            size_t p = 0;
            if (num_str[0] == '+' || num_str[0] == '-') {
                neg = (num_str[0] == '-');
                p = 1;
            }
            long val = 0;
            for (; p < num_str.size(); ++p) {
                char c = num_str[p];
                if (c < '0' || c > '9') {
                    throw std::runtime_error(std::string("Invalid header: cannot parse shape number: ") + num_str + " in: " + header);
                }
                val = val * 10 + (c - '0');
            }
            if (neg) val = -val;
            if (val < 0) {
                throw std::runtime_error(std::string("Negative shape dimension found: ") + num_str);
            }
            dims.push_back(static_cast<size_t>(val));
        }
        // skip until comma or end
        while (idx < shape_str.size() && shape_str[idx] != ',') ++idx;
        if (idx < shape_str.size() && shape_str[idx] == ',') ++idx;
    }

    if (dims.empty()) {
        // Zero-dim array? treat as 1
        dims.push_back(1);
    }

    size_t total = 1;
    for (size_t d : dims) {
        total *= d;
    }

    std::vector<float> data(total);
    file.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
    if (!file) {
        throw std::runtime_error("Failed to read data body: " + path);
    }
    return data;
}

int main() {
    // Initialize ACL resources first (as done in main.cpp)
    AclLiteResource aclDev;
    AclLiteError ret = aclDev.Init();
    if (ret != ACLLITE_OK) {
        std::cerr << "Failed to init ACL resources" << std::endl;
        return -1;
    }

    // Note: This test reads init_img.jpg / ini_box.txt and track_img.jpg

    Tracking tracker("./model/mixformerv2_online_small_bs1.om");
    if (tracker.InitModel() != 0) {
        std::cerr << "Failed to init model" << std::endl;
        return -1;
    }

    try {
        // Read init image and bbox (written by mixformerv2_om::init())
        cv::Mat init_img = cv::imread("init_img.jpg");
        if (init_img.empty()) {
            std::cerr << "Failed to read init_img.jpg or file not found" << std::endl;
            return -1;
        }

        // Parse ini_box.txt. Expected one line with: x0 y0 x1 y1 [class_id]
        // Only x0,y0,x1,y1 (and optional class_id) are read from file. w,h,cx,cy are recomputed.
        DrOBB init_box;
        init_box.score = 1.0f;
        init_box.initScore = init_box.score;
        init_box.class_id = 0;
        {
            std::ifstream bbox_file("ini_box.txt");
            if (!bbox_file) {
                std::cerr << "Failed to open ini_box.txt" << std::endl;
                return -1;
            }
            std::string line;
            std::getline(bbox_file, line);
            std::istringstream ss(line);
            float x0=0,y0=0,x1=0,y1=0; int class_id=0;
            if (!(ss >> x0 >> y0 >> x1 >> y1)) {
                std::cerr << "Invalid ini_box.txt format. Expected at least 'x0 y0 x1 y1'" << std::endl;
                return -1;
            }
            // Read any remaining tokens; support either formats:
            // - x0 y0 x1 y1
            // - x0 y0 x1 y1 class_id
            // - x0 y0 x1 y1 w h cx cy class_id
            std::vector<std::string> rem;
            std::string tok;
            while (ss >> tok) rem.push_back(tok);
            if (!rem.empty()) {
                // If there are multiple tokens, assume class_id is the last value
                int tmp = 0;
                if (!try_parse_int(rem.back(), tmp)) {
                    class_id = 0; // fallback
                } else {
                    class_id = tmp;
                }
            }
            // Ensure coordinates are ordered left-top -> right-bottom
            if (x1 < x0) std::swap(x0, x1);
            if (y1 < y0) std::swap(y0, y1);
            // Recompute width/height and center
            float w = x1 - x0;
            float h = y1 - y0;
            float cx = x0 + 0.5f * w;
            float cy = y0 + 0.5f * h;
            init_box.box.x0 = x0;
            init_box.box.y0 = y0;
            init_box.box.x1 = x1;
            init_box.box.y1 = y1;
            init_box.box.w = w;
            init_box.box.h = h;
            init_box.box.cx = cx;
            init_box.box.cy = cy;
            init_box.class_id = class_id;
        }

        // Initialize tracker with the image and bbox
        if (tracker.init(init_img, init_box) != 0) {
            std::cerr << "Failed to init tracker with init_img.jpg and ini_box.txt" << std::endl;
            return -1;
        }

        // Load track image and run tracker
        cv::Mat track_img = cv::imread("track_img.jpg");
        if (track_img.empty()) {
            std::cerr << "Failed to read track_img.jpg or file not found" << std::endl;
            return -1;
        }

        const DrOBB &tracked = tracker.track(track_img);
        std::cout << "Track Result: x0=" << tracked.box.x0 << " y0=" << tracked.box.y0 << " x1=" << tracked.box.x1 << " y1=" << tracked.box.y1 << " w=" << tracked.box.w << " h=" << tracked.box.h << " cx=" << tracked.box.cx << " cy=" << tracked.box.cy << " score=" << tracked.score << " class_id=" << tracked.class_id << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
