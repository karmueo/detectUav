#include "mixformerv2_om/mixformerv2_om.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

std::vector<float> load_npy(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    char header[128];
    file.read(header, 128);
    if (!file) {
        throw std::runtime_error("Failed to read header");
    }

    // Simple parsing: find 'shape': (size,)
    std::string h(header);
    size_t pos = h.find("'shape': (");
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid header: no shape found");
    }
    pos += 10;
    size_t end = h.find(')', pos);
    if (end == std::string::npos) {
        throw std::runtime_error("Invalid header: shape not closed");
    }
    std::string shape_str = h.substr(pos, end - pos);
    size_t size = std::stoul(shape_str);

    std::vector<float> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    if (!file) {
        throw std::runtime_error("Failed to read data");
    }

    return data;
}

int main() {
    // Expected input sizes (from model definition)
    const size_t expected_template_size = 1 * 3 * 112 * 112;        // 37632
    const size_t expected_online_template_size = 1 * 3 * 112 * 112; // 37632
    const size_t expected_search_size = 1 * 3 * 224 * 224;          // 150528

    MixformerV2OM tracker("model/mixformerv2_online_small.om");
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
