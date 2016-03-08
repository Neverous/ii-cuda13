#include "defines.h"
#include "simulation.h"

#include <cassert>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <limits>
#include <vector>
#include <algorithm>

#include <QtDebug>

#include "local.h"
#include "graph/graph.h"
#include "car.h"

using namespace std;
using namespace trafficsim;

Simulation::Simulation(Local &_local, QObject *_parent/* = nullptr*/)
:QThread(_parent)
,local(_local)
,state(false)
#ifdef __CPU__
,cpuworker()
#endif // __CPU__

#ifdef __CUDA__
,cudaworker()
#endif // __CUDA__
{
    setObjectName("Simulation");
#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
    {
        cpuworker[c].local = &_local;
        cpuworker[c].start();
    }
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.local = &_local;
    cudaworker.start();
#endif // __CUDA__
}

Simulation::~Simulation(void)
{
    stop();
}

void Simulation::stop(void)
{
    state = false;
    wait();

#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].stop();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.stop();
#endif // __CUDA__
}

void Simulation::run(void)
{
    state = true;
    while(state)
    {
        sync.restart();

        // recreate queue
        recreateQueues();

#ifdef __CPU__
        // Allocate cars to threads
        allocateCars();
#endif // __CPU__

        // Replenish already finished cars
        replenishCars();

        // Routing
        //routeCars();

        // Simulation
        simulateCars();

        // update car status
        updateCars();

        using namespace chrono;
        uint64_t elapsed = sync.restart();
        if(elapsed <= 1000.0 / SIMULATION_FPS)
            this_thread::sleep_for(milliseconds(static_cast<unsigned int>(1000.0 / SIMULATION_FPS - elapsed)));

        else
            qWarning() << "Simulation step took" << elapsed << "ms";
    }

    qDebug() << "Finished";
#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].stop();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.stop();
#endif // __CUDA__
}

inline
void Simulation::replenishCars(void)
{
#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].replenish();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.replenish();
#endif // __CUDA__

#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].join();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.join();
#endif // __CUDA__
}

#ifdef __CPU__
inline
void Simulation::allocateCars(void)
{
    int32_t     workers = 0;
    uint32_t    w       = 0;
    uint32_t    current = 0;

    // COUNT ELAPSED TIMES
    uint64_t elapsed = sumElapsedTimes(workers);

    assert(workers > 0);
    vector<int32_t> count(workers);
    int32_t         left = distributeCars(count, workers, elapsed);

    if(left < 0)
    {
        uint32_t sum = 0;
        for(int32_t _w = 0; _w < workers; ++ _w)
            sum += count[_w] *= 1.0f * local.car.size() / (local.car.size() - left);

        left = local.car.size() - sum;
    }

    assert(left >= 0);
    // FIX REMAINDER
    if(left > 0)
    {
        int32_t diff = max(1, (left + workers - 1) / workers);
        for(int32_t _w = 0; _w < workers && left; ++ _w)
        {
            left        -= diff;
            count[_w]   += diff;
            if(diff > left) diff = left;
        }
    }

    assert(left == 0);

    // APPLY CARS TO WORKERS
    w = 0;
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
    {
        cpuworker[c].begin  = current;
        cpuworker[c].end    = current += count[w ++];
        qDebug() << "CPUWorker" << c << "got" << cpuworker[c].end - cpuworker[c].begin << "cars" << cpuworker[c].elapsed;
    }

#ifdef __CUDA__
    cudaworker.begin  = current;
    cudaworker.end    = current += count[w ++];
    qDebug() << "CUDAWorker" << "got" << cudaworker.end - cudaworker.begin << "cars" << cudaworker.elapsed;
#endif // __CUDA__

    assert(current == local.car.size());
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].allocate();

#ifdef __CUDA__
    cudaworker.allocate();
#endif // __CUDA__

    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].join();

#ifdef __CUDA__
    cudaworker.join();
#endif // __CUDA__
}

inline
uint64_t Simulation::sumElapsedTimes(int32_t &workers)
{
    uint64_t elapsed = 0;

    workers += CPU_WORKERS;
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
    {
        assert(cpuworker[c].elapsed > 0);
        elapsed += cpuworker[c].elapsed;
    }

#ifdef __CUDA__
    ++ workers;
    assert(cudaworker.elapsed > 0);
    elapsed += cudaworker.elapsed;
#endif // __CUDA__

    return elapsed;
}

inline
int32_t Simulation::distributeCars(vector<int32_t> &distribution, int32_t workers, uint64_t elapsed)
{
    int32_t left    = local.car.size();
    int32_t w       = 0;
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        left -= distribution[w ++] = elapsed * (cpuworker[c].end - cpuworker[c].begin) / workers / cpuworker[c].elapsed;

#ifdef __CUDA__
    left -= distribution[w ++] = elapsed * (cudaworker.end - cudaworker.begin) / workers / cudaworker.elapsed;
#endif // __CUDA__

    return left;
}

#endif // __CPU__
/*
inline
void Simulation::routeCars(void)
{
#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].route();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.route();
#endif // __CUDA__

#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].join();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.join();
#endif // __CUDA__
}
*/
inline
void Simulation::simulateCars(void)
{
#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].simulate();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.simulate();
#endif // __CUDA__

#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].join();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.join();
#endif // __CUDA__
}

inline
void Simulation::updateCars(void)
{
#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].update();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.update();
#endif // __CUDA__

#ifdef __CPU__
    for(uint32_t c = 0; c < CPU_WORKERS; ++ c)
        cpuworker[c].join();
#endif // __CPU__

#ifdef __CUDA__
    cudaworker.join();
#endif // __CUDA__
}

inline
void Simulation::recreateQueues(void)
{
    vector<uint32_t> count(local.graph.edges, 0);
    uint32_t sum = 0;
    for(uint32_t c = 0; c < local.car.size(); ++ c)
        if(local.car[c].destination != numeric_limits<uint32_t>::max())
        {
            ++ count[local.car[c].route];
            ++ sum;
        }

    assert(local.car.size() >= sum);
    local.edgeQueue.resize(local.graph.edges);
    local.queue.resize(local.car.size());
    uint32_t _begin = 0;
    for(uint32_t e = 0; e < local.graph.edges; ++ e)
    {
        local.edgeQueue[e]  = _begin;
        _begin += count[e];
    }

    for(uint32_t c = 0; c < local.car.size(); ++ c)
        if(local.car[c].destination != numeric_limits<uint32_t>::max())
            local.queue[local.edgeQueue[local.car[c].route] + (-- count[local.car[c].route])] = c;
}
