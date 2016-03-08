#include "defines.h"
#include "worker.h"

#include <cassert>
#include <QDebug>
#include <thread>
#include <chrono>

#include "local.h"

using namespace std;
using namespace chrono;
using namespace trafficsim;
using namespace trafficsim::cpu;

Worker::Worker(QObject *_parent/* = nullptr*/)
:QThread(_parent)
,local(nullptr)
,begin(0)
,end(0)
,elapsed(1)
,state(1)
,sync()
,randSeed((uint64_t) this)
{
    setObjectName("CPUWorker");
}

Worker::~Worker(void)
{
    stop();
}

void Worker::stop(void)
{
    state = 0;
    wait();
}

void Worker::allocate(void)
{
    state = 2;
}

void Worker::replenish(void)
{
    state = 3;
}

void Worker::route(void)
{
    state = 4;
}

void Worker::simulate(void)
{
    state = 5;
}

void Worker::update(void)
{
    state = 6;
}

void Worker::join(void)
{
    while(state > 1)
        sleep(0);

    return;
}

void Worker::run(void)
{
    while(state)
    {
        while(state == 1)
            sleep(0);

        assert(state != 1);
        sync.restart();
        switch(state)
        {
            case 0:
                continue;

            case 2:
                elapsed = 1;
                doAllocate();
                break;

            case 3:
                doReplenish();
                break;

            case 4:
                doRoute();
                break;

            case 5:
                doSimulate();
                break;

            case 6:
                doUpdate();
                break;
        }

        elapsed += sync.restart();
        state = 1;
    }

    qDebug() << "Finished";
}

inline
void Worker::doAllocate(void)
{
    memcpy(&local->prev[begin], &local->car[begin], (end - begin) * sizeof(Car));
}

inline
void Worker::doReplenish(void)
{
    for(uint32_t c = begin; c < end; ++ c)
    {
        auto &car = local->car[c];
        if(car.destination == numeric_limits<uint32_t>::max())
        {
            replenishCar(c);
            local->prev[c] = car;
        }
    }
}

inline
void Worker::replenishCar(uint32_t c)
{
    Car &car = local->car[c];
    car.destination         = fast_rand() % local->graph.nodes;
    const graph::Node &from = local->graph.node[car.destination];
    car.position            = vec2(from.x, from.y);
    assert(local->prev[c].destination == numeric_limits<uint32_t>::max());
    if(nextEdge(c) < 2)
        car.destination = numeric_limits<uint32_t>::max();

    car.speed = 0.0f;
}

inline
uint8_t Worker::nextEdge(uint32_t c)
{
    Car &car = local->car[c];
    const graph::Node &from = local->graph.node[car.destination];
    uint32_t    _begin  = from.edge;
    uint32_t    _end    = car.destination < local->graph.nodes - 1 ? local->graph.node[car.destination + 1].edge : local->graph.nodeEdges;

    if(_begin == _end) // no edges connected to the node (?)
        return 0;

    uint32_t    _e       = _begin + (fast_rand() % (_end - _begin));
    uint32_t    block   = 0;
    uint32_t    _route  = 0;
    do
    {
        _route   = local->graph.nodeEdge[_e];
        ++ _e;
        if(_e == _end)
            _e = _begin;
    }
    while(++ block < 10 && (_route == car.route || (local->graph.edge[_route].oneway() && local->graph.edge[_route].to == car.destination)));

    if(block == 10 || (local->graph.edge[_route].oneway() && local->graph.edge[_route].to == car.destination) || _route == car.route)
        return 0; // edge choice failed

    const graph::Edge &edge = local->graph.edge[_route];
    uint32_t destination = car.destination == edge.to ? edge.from : edge.to;
    const graph::Node &to     = local->graph.node[destination];
    vec2 direction       = vec2(to.x - from.x, to.y - from.y).normalized();
    if(direction.length() <= 0.9f || 1.1f <= direction.length())
        return 0; // invalid edge (prob len == 0)

    uint8_t lane = fast_rand() % edge.lanes();
    vec2 pos = car.position;
    car.position    = vec2(from.x, from.y);

    for(uint32_t q = local->edgeQueue[_route], r = _route < local->graph.edges - 1 ? local->edgeQueue[_route + 1] : local->queue.size(); q < r; ++ q)
    {
        uint32_t &d = local->queue[q];
        if(d == c) continue;
        Car &test = local->car[d];
        if(test.lane != lane) continue;
        if(test.direction != direction) continue;
        if(test.distanceSquared(vec2(to.x, to.y)) >= car.distanceSquared(vec2(to.x, to.y))) continue;
        float dist = car.distance(test.position);
        if(dist < CAR_LENGTH) continue;

        if(dist <= CAR_LENGTH + 2.0f)
        {
            car.speed = 0.0f;
            car.position = pos;
            return 1;
        }
    }

    car.route       = _route;
    car.direction   = direction;
    car.lane        = lane;
    car.destination = destination;

    assert(0.9f <= car.direction.length() && car.direction.length() <= 1.1f);
    const vec2 normal  = vec2(from.y - to.y, to.x - from.x).normalized();
    car.position -= normal * graph::LANE_WIDTH * (edge.lanes() / (edge.oneway() + 1.0f) - 0.5f - car.lane);
    return 2;
}

inline
void Worker::doRoute(void)
{
}

inline
void Worker::doSimulate(void)
{
    for(uint32_t c = begin; c < end; ++ c)
    {
        auto &car = local->car[c];
        const auto &prev = local->prev[c];
        if(prev.destination != numeric_limits<uint32_t>::max())
        {
            assert(0.9f <= prev.direction.length() && prev.direction.length() <= 1.1f);
            const graph::Edge &edge = local->graph.edge[prev.route];
            const float limit = edge.maxSpeed() ? edge.maxSpeed() * 5.0f / 18.0f / SIMULATION_FPS : 14.0f / SIMULATION_FPS;

            const graph::Node &to = local->graph.node[prev.destination];

            car.speed = min(limit, prev.speed + CAR_ACCELERATION / SIMULATION_FPS);
#ifdef CROSS_COLLISIONS
            if(vec2(to.x - car.position.x, to.y - car.position.y).length() < 4.0f * CAR_LENGTH)
                for(uint32_t e = to.edge, f = prev.destination < local->graph.nodes - 1 ? local->graph.node[prev.destination + 1].edge : local->graph.nodeEdges; car.speed > 0.0f && e < f; ++ e)
                    for(uint32_t q = local->edgeQueue[local->graph.nodeEdge[e]], r = local->graph.nodeEdge[e] < local->graph.edges - 1 ? local->edgeQueue[local->graph.nodeEdge[e] + 1] : local->queue.size(); q < r; ++ q)
                    {
                        uint32_t &d = local->queue[q];
                        if(d == c) continue;
                        Car &test = local->prev[d];
                        float dist = car.distance(test.position);
                        if(dist > 2.0f * CAR_LENGTH) continue;
                        if(dist < CAR_LENGTH) continue;
                        if(!car.collide(test)) continue;
                        if(d < c)
                            car.speed = max(0.0f, min(car.speed, sqrtf(dist) - 1.0f));

                        if(dist <= CAR_LENGTH + 2.0f)
                        {
                            car.speed = 0.0f;
                            break;
                        }
                    }

            else
#endif // CROSS_COLLISIONS
                for(uint32_t q = local->edgeQueue[car.route], r = car.route < local->graph.edges - 1 ? local->edgeQueue[car.route + 1] : local->queue.size(); q < r; ++ q)
                {
                    uint32_t &d = local->queue[q];
                    if(d == c) continue;
                    Car &test = local->prev[d];
                    if(test.lane != car.lane) continue;
                    if(test.direction != car.direction) continue;
                    if(test.distanceSquared(vec2(to.x, to.y)) > car.distanceSquared(vec2(to.x, to.y))) continue;
                    if(test.distanceSquared(vec2(to.x, to.y)) == car.distanceSquared(vec2(to.x, to.y)) && d > c) continue;
                    float dist = car.distance(test.position);
                    if(d < c)
                        car.speed = max(0.0f, min(car.speed, sqrtf(dist) - 1.0f));

                    if(dist <= CAR_LENGTH + 2.0f)
                    {
                        car.speed = 0.0f;
                        break;
                    }
                }

            if(car.speed)
            {
                car.position = prev.position + prev.direction * car.speed;
                float before = vec2(prev.position.x - to.x, prev.position.y - to.y).lengthSquared();
                float now = vec2(car.position.x - to.x, car.position.y - to.y).lengthSquared();
                if(before < now)
                    car.position = prev.position;

                if((before < now || now <= CAR_LENGTH / 4.0f) && !nextEdge(c))
                    car.destination = numeric_limits<uint32_t>::max();
            }
        }
    }
}

inline
void Worker::doUpdate(void)
{
}

inline
uint32_t Worker::fast_rand(void)
{
    randSeed = (214013 * randSeed + 2531011);
    return (randSeed >> 16) & 0x7FFF;
}
