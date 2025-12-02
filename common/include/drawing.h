#ifndef YOLOV3_COCO_DETECTION_PICTURE_WITH_FREETYPE_INC_DRAWING_H
#define YOLOV3_COCO_DETECTION_PICTURE_WITH_FREETYPE_INC_DRAWING_H

// #include <algorithm>
#include <cmath>
// #include <iostream>
// #include <stdint.h>
// #include <string>
// #include "freetype_helper.h"
#include "AclLiteType.h"

class YUVColor
{
  public:
    YUVColor(uint8_t y, uint8_t u, uint8_t v) : y(y), u(u), v(v) {}
    uint8_t y;
    uint8_t u;
    uint8_t v;
};

void DrawText(
    ImageData         &image,
    int                x,
    int                y,
    const std::string &text,
    const YUVColor    &color,
    int                fontSize = 24,
    float              alpha = 1.0f);
void DrawRect(
    ImageData      &image,
    int             x1,
    int             y1,
    int             x2,
    int             y2,
    const YUVColor &color,
    int             lineWidth);

#endif