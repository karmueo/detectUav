#ifndef FREETYPE_HELPER_H
#define FREETYPE_HELPER_H

#include <string>
#include <vector>
#include <ft2build.h>
#include FT_FREETYPE_H

struct TextBitmap {
    int width;
    int height;
    std::vector<uint8_t> data; // grayscale alpha values, 0-255
};

TextBitmap RenderText(const std::string& text, int fontSize);

#endif // FREETYPE_HELPER_H