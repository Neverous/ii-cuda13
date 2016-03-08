#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include <QThread>
#include <QElapsedTimer>

#include "local.h"
#include "car.h"

#ifdef __CPU__
    #include "cpu/worker.h"
#endif // __CPU__

#ifdef __CUDA__
    #include "cuda/worker.h"
#endif // __CUDA__

namespace trafficsim
{

class Simulation: public QThread
{
    Q_OBJECT

    Local &local;
    bool state;

#ifdef __CPU__
    cpu::Worker     cpuworker[CPU_WORKERS];
#endif // __CPU__

#ifdef __CUDA__
    cuda::Worker    cudaworker;
#endif // __CUDA__

    QElapsedTimer   sync;

    public:
        Simulation(Local &_local, QObject *_parent = nullptr);
        ~Simulation(void);
        void stop(void);

    protected:
        void run(void);

    private:
        void replenishCars(void);
#ifdef __CPU__
        void allocateCars(void);
        uint64_t sumElapsedTimes(int32_t &workers);
        int32_t distributeCars(vector<int32_t> &distribution, int32_t workers, uint64_t elapsed);
#endif // __CPU__
        //void routeCars(void);
        void simulateCars(void);
        void updateCars(void);

        void recreateQueues(void);
}; // class Simulation

} // namespace trafficsim

#endif // __SMIULATION_H__
