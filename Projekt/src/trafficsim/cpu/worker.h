#ifndef __CPU_WORKER_H__
#define __CPU_WORKER_H__

#include <QThread>
#include <QElapsedTimer>

#include "local.h"

namespace trafficsim
{

namespace cpu
{

using namespace std;

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
        uint32_t            randSeed;

    public:
        Worker(QObject *_parent = nullptr);
        ~Worker(void);
        void stop(void);

        void allocate(void);
        void replenish(void);
        void route(void);
        void simulate(void);
        void update(void);
        void join(void);

    protected:
        void run(void);

    private:
        void doAllocate(void);
        void doReplenish(void);
        void replenishCar(uint32_t c);
        uint8_t nextEdge(uint32_t c);
        void doRoute(void);
        void doSimulate(void);
        void doUpdate(void);

        uint32_t fast_rand(void);
}; // class Worker

} // namespace cpu

} // namespace trafficsim

#endif // __CPU_WORKER_H__
