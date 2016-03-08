#ifndef __cuda__
    #define __cuda__ __device__
#endif // __cuda__

#include "graph/graph.h"
#include "trafficsim/car.h"

using namespace trafficsim;

/* HELPER FUNCTIONS */
void cudaWorkerInitialize(graph::Graph &_graph);
void cudaWorkerCopyCars(Car *car, uint32_t count, uint32_t begin, uint32_t end);
void cudaWorkerCopyQueues(uint32_t *queue, uint32_t queueSize, uint32_t *edgeQueue, uint32_t edgeQueueSize);
void cudaWorkerReplenishCars(void);
//void cudaWorkerRoute(void);
void cudaWorkerSimulate(void);
void cudaWorkerUpdate(Car *begin, uint32_t count);
void cudaWorkerFinished(void);
