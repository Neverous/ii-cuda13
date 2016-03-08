#include "defines.h"
#include "worker.h"

#include <cassert>
#include <QDebug>
#include <thread>
#include <chrono>

using namespace std;
using namespace chrono;
using namespace trafficsim;
using namespace trafficsim::cuda;

Worker::Worker(QObject *_parent/* = nullptr*/)
:QThread(_parent)
,local(nullptr)
,begin(0)
,end(0)
,elapsed(1)
,state(1)
,sync()
{
    setObjectName("CUDAWorker");
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
/*
void Worker::route(void)
{
    state = 4;
}
*/
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
    cudaWorkerInitialize(local->graph);
#ifndef __CPU__
    begin   = 0;
    end     = local->car.size();
    cudaWorkerCopyCars(&local->car[0], local->car.size(), begin, end);
#endif // __CPU__
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

            /*case 4:
                doRoute();
                break;*/

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

    cudaWorkerFinished();
    qDebug() << "Finished";
}

inline
void Worker::doAllocate(void)
{
    if(end == begin)
        return;

#ifdef __CPU__
    cudaWorkerCopyCars(&local->car[0], local->car.size(), begin, end);
#endif // __CPU__
}

inline
void Worker::doReplenish(void)
{
    if(end == begin)
        return;

    cudaWorkerCopyQueues(&local->queue[0], local->queue.size(), &local->edgeQueue[0], local->edgeQueue.size());
    cudaWorkerReplenishCars();
}

/*
inline
void Worker::doRoute(void)
{
    if(end == begin)
        return;

    cudaWorkerRoute();
}
*/
inline
void Worker::doSimulate(void)
{
    if(end == begin)
        return;

    cudaWorkerSimulate();
}

inline
void Worker::doUpdate(void)
{
    if(end == begin)
        return;

    cudaWorkerUpdate(&local->car[begin], end - begin);
}
