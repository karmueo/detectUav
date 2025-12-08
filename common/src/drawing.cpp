#include <algorithm>
#include <cmath>
// #include <iostream>
// #include <stdint.h>
// #include <string>
// #include <stdio.h>
#include "drawing.h"
#include "freetype_helper.h"

// Helper functions for YUV <-> RGB conversion
struct RGBColor {
    uint8_t r, g, b;
};

RGBColor YUVToRGB(uint8_t y, uint8_t u, uint8_t v) {
    int r = y + 1.402 * (v - 128);
    int g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128);
    int b = y + 1.772 * (u - 128);
    r = std::max(0, std::min(255, r));
    g = std::max(0, std::min(255, g));
    b = std::max(0, std::min(255, b));
    return {static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b)};
}

YUVColor RGBToYUV(uint8_t r, uint8_t g, uint8_t b) {
    int y = 0.299 * r + 0.587 * g + 0.114 * b;
    int u = -0.169 * r - 0.331 * g + 0.5 * b + 128;
    int v = 0.5 * r - 0.419 * g - 0.081 * b + 128;
    y = std::max(0, std::min(255, y));
    u = std::max(0, std::min(255, u));
    v = std::max(0, std::min(255, v));
    return {static_cast<uint8_t>(y), static_cast<uint8_t>(u), static_cast<uint8_t>(v)};
}

// Get YUV pixel from image
YUVColor GetPixel(const ImageData& image, int x, int y) {
    uint8_t y_val = *(image.data.get() + y * image.width + x);
    uint8_t* uv_offset = image.data.get() + image.width * image.height + (y / 2) * image.width + (x / 2) * 2;
    uint8_t u = uv_offset[0];
    uint8_t v = uv_offset[1];
    return {y_val, u, v};
}

void SetPixel(ImageData &image, int x, int y, const YUVColor &color)
{
    *(image.data.get() + y * image.width + x) = color.y;
    uint8_t *uv_offset = image.data.get() + image.width * image.height +
                         (y / 2) * image.width + x / 2 * 2;
    uv_offset[0] = color.u;
    uv_offset[1] = color.v;
}

void DrawText(
    ImageData         &image,
    int                x,
    int                y,
    const std::string &text,
    const YUVColor    &color,
    int                fontSize,
    float              alpha)
{
    TextBitmap bitmap = RenderText(text, fontSize);
    RGBColor textRGB = YUVToRGB(color.y, color.u, color.v); // Convert text color to RGB

    for (int by = 0; by < bitmap.height; ++by) {
        for (int bx = 0; bx < bitmap.width; ++bx) {
            int px = x + bx;
            int py = y + by;
            if (px >= 0 && px < (int)image.width && py >= 0 && py < (int)image.height) {
                uint8_t alpha_val = bitmap.data[by * bitmap.width + bx];
                if (alpha_val > 0) {
                    YUVColor currentYUV = GetPixel(image, px, py);
                    RGBColor currentRGB = YUVToRGB(currentYUV.y, currentYUV.u, currentYUV.v);
                    // Blend
                    uint8_t r = alpha * alpha_val / 255.0f * textRGB.r + (1 - alpha * alpha_val / 255.0f) * currentRGB.r;
                    uint8_t g = alpha * alpha_val / 255.0f * textRGB.g + (1 - alpha * alpha_val / 255.0f) * currentRGB.g;
                    uint8_t b = alpha * alpha_val / 255.0f * textRGB.b + (1 - alpha * alpha_val / 255.0f) * currentRGB.b;
                    YUVColor newYUV = RGBToYUV(r, g, b);
                    SetPixel(image, px, py, newYUV);
                }
            }
        }
    }
}

void DrawRect(
    ImageData      &image,
    int             x1,
    int             y1,
    int             x2,
    int             y2,
    const YUVColor &color,
    int             lineWidth)
{
    // Draw an open rectangle using corner segments instead of solid edges.
    if (x1 > x2)
    {
        std::swap(x1, x2);
    }

    if (y1 > y2)
    {
        std::swap(y1, y2);
    }

    int i, j;
    int iBound, jBound;
    int iStart, jStart;
    int width = (int)image.width;
    int height = (int)image.height;

    auto DrawBlock = [&](int xs, int ys, int xe, int ye)
    {
        if (xs > xe || ys > ye)
        {
            return;
        }

        xs = std::max(0, xs);
        ys = std::max(0, ys);
        xe = std::min(width - 1, xe);
        ye = std::min(height - 1, ye);

        for (int row = ys; row <= ye; ++row)
        {
            for (int col = xs; col <= xe; ++col)
            {
                SetPixel(image, col, row, color);
            }
        }
    };

    int box_width = x2 - x1 + 1;
    int box_height = y2 - y1 + 1;
    int min_dim = std::min(box_width, box_height);
    int corner_len = std::max(lineWidth * 3, min_dim / 4);
    corner_len = std::min(corner_len, min_dim);

    // Top corners
    DrawBlock(x1, y1, x1 + corner_len - 1, y1 + lineWidth - 1);
    DrawBlock(x2 - corner_len + 1, y1, x2, y1 + lineWidth - 1);
    // Bottom corners
    DrawBlock(x1, y2 - lineWidth + 1, x1 + corner_len - 1, y2);
    DrawBlock(x2 - corner_len + 1, y2 - lineWidth + 1, x2, y2);
    // Left vertical corners
    DrawBlock(x1, y1, x1 + lineWidth - 1, y1 + corner_len - 1);
    DrawBlock(x1, y2 - corner_len + 1, x1 + lineWidth - 1, y2);
    // Right vertical corners
    DrawBlock(x2 - lineWidth + 1, y1, x2, y1 + corner_len - 1);
    DrawBlock(x2 - lineWidth + 1, y2 - corner_len + 1, x2, y2);
}
