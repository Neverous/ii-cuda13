#ifndef __CUDA_WORKER_H__
#define __CUDA_WORKER_H__

#include <QThread>
#include <QElapsedTimer>

#include "graph/graph.h"
#include "trafficsim/car.h"
#include "worker.cuh"
#include "local.h"

namespace trafficsim
{

namespace cuda
{

class Worker: public QThread
{
    Q_OBJECT

    public:
        Local       *local;
        uint32_t    begin;
        uint32_t    end;
        uint64_t    elapsed;

    private:
        uint8_t             state;
        QElapsedTimer       sync;

    public:
        Worker(QObject *_parent = nullptr);
        ~Worker(void);
        void stop(void);

        void allocate(void);
        void replenish(void);
        //void route(void);
        void simulate(void);
        void update(void);
        void join(void);

    protected:
        void run(void);

    private:
        void doAllocate(void);
        void doReplenish(void);
        //void doRoute(void);
        void doSimulate(void);
        void doUpdate(void);
}; // class Worker

} // namespace cuda

} // namespace trafficsim

#endif // __CUDA_WORKER_H__
