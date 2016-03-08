#include "fraktal.h"
#include "fraktal_ispc.h"

using namespace ispc;

void ISPC_Atraktor(char *image, int width, int height, float s)
{
    fraktal_ispc((int8_t *) image, width, height, s);
}
