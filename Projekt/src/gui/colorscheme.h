#ifndef __COLORSCHEME_H__
#define __COLORSCHEME_H__

#include <QColor>
#include "GLObjects.h"

enum Brightness
{
    LIGHT   = 0,
    NORMAL  = 1,
    DARK    = 2,
}; // enum Brightness

// TANGO
namespace Colorscheme
{
    //static int Yellow[]   = {0xFCE94F, 0xEDD400, 0xC4A000};
    //static int Orange[]   = {0xFCAF3E, 0xF57900, 0xCE5C00};
    //static int Brown[]    = {0xe9b96e, 0xc17d11, 0x8f5902};
    //static int Green[]    = {0x8ae234, 0x73d216, 0x4e9a06};
    //static int Blue[]     = {0x729fcf, 0x3465a4, 0x204a87};
    //static int Violet[]   = {0xad7fa8, 0x75507b, 0x5c3566};
    //static int Red[]      = {0xef2929, 0xcc0000, 0xa40000};
    //static int White[]    = {0xeeeeec, 0xd3d7cf, 0xbabdb6};
    static int Black[]    = {0x888a85, 0x555753, 0x2e3436};
}; // namespace Colorscheme

inline
static const QColor QCOLOR(unsigned int rgba)
{
    return QColor((rgba >> 16) & 0xFF, (rgba >> 8) & 0xFF, rgba & 0xFF, (rgba >> 24) & 0xFF);
}

inline
static const GLColor GLCOLOR(unsigned int rgba)
{
    return GLColor(((rgba >> 16) & 0xFF) / 255.0f, ((rgba >> 8) & 0xFF) / 255.0f, (rgba & 0xFF) / 255.0f);
}

#endif // __COLORSCHEME_H__
