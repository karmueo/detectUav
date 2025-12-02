#include "freetype_helper.h"
#include <iostream>
#include <codecvt>
#include <locale>

TextBitmap RenderText(const std::string& text, int fontSize) {
    FT_Library library;
    FT_Face face;
    TextBitmap bitmap;

    if (FT_Init_FreeType(&library)) {
        std::cerr << "Could not init FreeType library" << std::endl;
        return bitmap;
    }

    // Load a default font, e.g., /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
    // You may need to adjust the path
    if (FT_New_Face(library, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0, &face)) {
        std::cerr << "Could not load font" << std::endl;
        FT_Done_FreeType(library);
        return bitmap;
    }

    FT_Set_Pixel_Sizes(face, 0, fontSize);

    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> cvt;
    std::u32string utf32_str = cvt.from_bytes(text);

    int totalWidth = 0;
    int maxAscent = 0;
    int maxDescent = 0;
    for (char32_t u32char : utf32_str) {
        if (FT_Load_Char(face, u32char, FT_LOAD_RENDER)) {
            continue;
        }
        totalWidth += face->glyph->advance.x >> 6;
        int ascent = face->glyph->bitmap_top;
        int descent = face->glyph->bitmap.rows - face->glyph->bitmap_top;
        if (ascent > maxAscent) maxAscent = ascent;
        if (descent > maxDescent) maxDescent = descent;
    }

    int baseline_y = maxAscent;
    bitmap.width = totalWidth;
    bitmap.height = maxAscent + maxDescent;
    bitmap.data.resize(totalWidth * bitmap.height, 0);

    int x = 0;
    for (char32_t u32char : utf32_str) {
        if (FT_Load_Char(face, u32char, FT_LOAD_RENDER)) {
            continue;
        }
        FT_Bitmap& ftBitmap = face->glyph->bitmap;
        int glyph_x = x + face->glyph->bitmap_left;
        int glyph_y = baseline_y - face->glyph->bitmap_top;
        for (int by = 0; by < static_cast<int>(ftBitmap.rows); ++by) {
            for (int bx = 0; bx < static_cast<int>(ftBitmap.width); ++bx) {
                int px = glyph_x + bx;
                int py = glyph_y + by;
                if (px >= 0 && px < totalWidth && py >= 0 && py < bitmap.height) {
                    bitmap.data[py * totalWidth + px] = ftBitmap.buffer[by * ftBitmap.width + bx];
                }
            }
        }
        x += face->glyph->advance.x >> 6;
    }

    FT_Done_Face(face);
    FT_Done_FreeType(library);
    return bitmap;
}