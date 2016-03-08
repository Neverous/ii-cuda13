/*
 * Elliptical Mercator
 * http://wiki.openstreetmap.org/wiki/Mercator#Elliptical_Mercator
 */
#ifndef __MERCATOR_H__
#define __MERCATOR_H__

#include <cmath>

namespace trafficsim
{

namespace projection
{

namespace mercator
{

static const float EQUATORIAL_RADIUS    = 6378137.0f;
static const float POLAR_RADIUS         = 6356752.3142f;
static const float RADIUS_RATIO         = POLAR_RADIUS / EQUATORIAL_RADIUS;
static const float ECCENT               = sqrtf(1.0f - RADIUS_RATIO * RADIUS_RATIO);
static const float COM                  = 0.5f * ECCENT;

inline
float lonToMet(float lon)
{
    return EQUATORIAL_RADIUS * lon * M_PI / 180.0f;
}

inline
float latToMet(float lat)
{
    lat = min(89.5f, max(lat, -89.5f));
    float phi       = lat * M_PI / 180.0f;
    float con       = ECCENT * sinf(phi);
    con = powf((1.0f - con) / (1.0f + con), COM);
    return 0.0f - EQUATORIAL_RADIUS * logf(tanf(0.5f * (M_PI * 0.5f - phi)) / con);
}

inline
float metToLon(float x)
{
    return 180.0f * x / EQUATORIAL_RADIUS / M_PI;
}

inline
float metToLat(float y)
{
    float ts    = expf(-y / EQUATORIAL_RADIUS);
    float phi   = M_PI_2 - 2.0f * atanf(ts);
    float dphi  = 1.0f;
    for(int i = 0; fabs(dphi) > 0.000000001f && i < 15; ++ i)
    {
        float con   = ECCENT * sinf(phi);
        dphi = M_PI_2 - 2.0f * atanf(ts * powf((1.0f - con) / (1.0f + con), COM)) - phi;
        phi += dphi;
    }

    return 180.0f * phi / M_PI;
}

} // namespace mercator

} // namespace projection

} // namespace trafficsim

#endif // __MERCATOR_H__
