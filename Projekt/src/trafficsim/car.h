#ifndef __CAR_H__
#define __CAR_H__

#include <cfloat>
#include "vec2.h"

#define MAX_SPEED   36.0

namespace trafficsim
{

const float CAR_WIDTH          = 1.8f;
const float CAR_LENGTH         = 4.8f;
const float CAR_ACCELERATION   = 1.0f;

struct Car
{
    vec2        direction;

    // m/s
    float       speed;

    // placement
    vec2        position;
    uint8_t     lane;

    // routing
    uint32_t    route;
    uint32_t    destination;

    __cuda__ float distance(const vec2 &point);
    __cuda__ float distanceSquared(const vec2 &point);
    __cuda__ bool collide(Car &car);
}; // struct Car [33]

inline
__cuda__ float Car::distance(const vec2 &point)
{
    return (point - position).length();
}

inline
__cuda__ float Car::distanceSquared(const vec2 &point)
{
    return (point - position).lengthSquared();
}

inline
__cuda__ bool Car::collide(Car &car)
{
    const vec2 normal(-direction.y, direction.x);
    const vec2 cnormal(-car.direction.y, car.direction.x);
    vec2 rect[2][4] = {{
        position + normal * CAR_WIDTH / 2.0f - direction * CAR_LENGTH / 4.0f,
        position + normal * CAR_WIDTH / 2.0f + direction * CAR_LENGTH * 2.0f,
        position - normal * CAR_WIDTH / 2.0f + direction * CAR_LENGTH * 2.0f,
        position - normal * CAR_WIDTH / 2.0f - direction * CAR_LENGTH / 4.0f,
    }, {
        car.position + cnormal * CAR_WIDTH / 2.0f - car.direction * CAR_LENGTH / 2.0f,
        car.position + cnormal * CAR_WIDTH / 2.0f + car.direction * CAR_LENGTH / 2.0f,
        car.position - cnormal * CAR_WIDTH / 2.0f + car.direction * CAR_LENGTH / 2.0f,
        car.position - cnormal * CAR_WIDTH / 2.0f - car.direction * CAR_LENGTH / 2.0f,
    },
    };

    for(uint32_t r = 0; r < 2; ++ r)
        for(uint32_t p = 0; p < 4; ++ p)
        {
            uint32_t q = (p + 1) % 4;
            const vec2 p1 = rect[r][p];
            const vec2 p2 = rect[r][q];

            const vec2 norm(p2.y - p1.y, p1.x - p2.x);

            float minA = FLT_MAX;
            float maxA = FLT_MIN;
            for(int c = 0; c < 4; ++ c)
            {
                float projected = norm.x * rect[0][c].x + norm.y * rect[0][c].y;
                if(projected < minA)
                    minA = projected;

                if(projected > maxA)
                    maxA = projected;
            }

            float minB = FLT_MAX;
            float maxB = FLT_MIN;
            for(int c = 0; c < 4; ++ c)
            {
                float projected = norm.x * rect[1][c].x + norm.y * rect[1][c].y;
                if(projected < minB)
                    minB = projected;

                if(projected > maxB)
                    maxB = projected;
            }

            if(maxA < minB || maxB < minA)
                return false;
        }

    return true;
}

} // namespace trafficsim

#endif // __CAR_H__
