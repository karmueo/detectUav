#include "mixformerv2_om/mixformerv2_om.h"
#include "AclLiteResource.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

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

    // Expected input sizes (from model definition)
    const size_t expected_template_size = 1 * 3 * 112 * 112;        // 37632
    const size_t expected_online_template_size = 1 * 3 * 112 * 112; // 37632
    const size_t expected_search_size = 1 * 3 * 224 * 224;          // 150528

    MixformerV2OM tracker("./model/mixformerv2_online_small_bs1.om");
    if (tracker.InitModel() != 0) {
        std::cerr << "Failed to init model" << std::endl;
        return -1;
    }

    try {
        // Load inputs
        auto template_data = load_npy("data/test_inputs/template_input.npy");
        auto online_template_data = load_npy("data/test_inputs/online_template_input.npy");
        auto search_data = load_npy("data/test_inputs/search_input.npy");

        // Check sizes
        if (template_data.size() != expected_template_size) {
            std::cerr << "Template size mismatch: expected " << expected_template_size << ", got " << template_data.size() << std::endl;
            return -1;
        }
        if (online_template_data.size() != expected_online_template_size) {
            std::cerr << "Online template size mismatch: expected " << expected_online_template_size << ", got " << online_template_data.size() << std::endl;
            return -1;
        }
        if (search_data.size() != expected_search_size) {
            std::cerr << "Search size mismatch: expected " << expected_search_size << ", got " << search_data.size() << std::endl;
            return -1;
        }

        // Set input data
        tracker.setInputTemplateData(template_data.data(), template_data.size());
        tracker.setInputOnlineTemplateData(online_template_data.data(), online_template_data.size());
        tracker.setInputSearchData(search_data.data(), search_data.size());

        // Run inference
        tracker.infer();

        // Get and print outputs
        std::cout << "Pred Boxes: ";
        auto boxes = tracker.getOutputPredBoxes();
        for (size_t i = 0; i < boxes.size() && i < 4; ++i) {
            std::cout << boxes[i] << " ";
        }
        std::cout << std::endl;
        
        auto scores = tracker.getOutputPredScores();
        if (!scores.empty()) {
            std::cout << "Pred Score: " << scores[0] << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
