#include "defines.h"
#include "worker.cuh"

#include <cstdio>

#define cudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line)
{
    if(code != cudaSuccess)
        fprintf(stderr,"CUDA: %s %s %d\n", cudaGetErrorString(code), file, line);
}

struct Cuda
{
    Car             *prev;
    uint32_t        cars;
    Car             *car;
    uint32_t        nodes;
    graph::Node     *node;
    uint32_t        edges;
    graph::Edge     *edge;
    uint32_t        connections;
    uint32_t        *connection;
    uint32_t        begin;
    uint32_t        end;
    uint32_t        *seed;
    uint32_t        blocks;

    uint32_t        queues;
    uint32_t        *queue;
    uint32_t        edgeQueues;
    uint32_t        *edgeQueue;
} cuda;

/*****************************************************************************/
/************************* DEVICE ********************************************/

__device__ uint32_t fast_rand(uint32_t *seed)
{
    *seed = (214013 * *seed + 2531011);
    return (*seed >> 16) & 0x7FFF;
}

__device__ uint8_t nextEdge(uint32_t *seed, Car *__car, uint32_t c, Car &car, uint32_t nodes, graph::Node *node, uint32_t edges, graph::Edge *edge, uint32_t connections, uint32_t *connection, uint32_t queues, uint32_t *queue, uint32_t edgeQueues, uint32_t *edgeQueue)
{
    const graph::Node &from = node[car.destination];
    uint32_t _begin = from.edge;
    uint32_t _end = car.destination < nodes - 1 ? node[car.destination + 1].edge : connections;

    if(_begin == _end) // random failed: no edges
        return 0;

    uint32_t    _e      = _begin + (fast_rand(seed + threadIdx.x) % (_end - _begin));
    uint32_t    block   = 0;
    uint32_t    _route  = 0;
    do
    {
        _route  = connection[_e];
        ++ _e;
        if(_e == _end)
            _e = _begin;
    }
    while(++ block < 10 && (_route == car.route || (edge[_route].oneway() && edge[_route].to == car.destination)));

    if(block == 10 || (edge[_route].oneway() && edge[_route].to == car.destination) || _route == car.route)
        return 0; // random failed: no edges

    const graph::Edge edg = edge[_route];
    uint32_t destination = car.destination == edg.to ? edg.from : edg.to;
    const graph::Node to     = node[destination];
    vec2 direction       = vec2(to.x - from.x, to.y - from.y).normalized();
    if(direction.length() <= 0.9f || 1.1f <= direction.length())
        return 0; // random failed: too short

    uint32_t lane = fast_rand(seed + threadIdx.x) % edg.lanes();
    vec2 pos = car.position;
    car.position = vec2(from.x, from.y);

    for(uint32_t q = edgeQueue[_route], r = _route < edges - 1 ? edgeQueue[_route + 1] : queues; q < r; ++ q)
    {
        uint32_t &d = queue[q];
        if(d == c) continue;
        Car &test = __car[d];
        if(test.lane != lane) continue;
        if(test.direction != direction) continue;
        if(test.distanceSquared(vec2(to.x, to.y)) >= car.distanceSquared(vec2(to.x, to.y))) continue;
        float dist = car.distance(test.position);
        if(dist < CAR_LENGTH && d < c) continue;

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

    const vec2 normal  = vec2(from.y - to.y, to.x - from.x).normalized();
    car.position -= normal * graph::LANE_WIDTH * (edg.lanes() / (edg.oneway() + 1.0f) - 0.5f - car.lane);
    return 2;
}


/*****************************************************************************/
/************************* KERNELS *******************************************/

__global__ void ReplenishCars(
    uint32_t    *seed,
    Car         *_car,
    uint32_t    begin,
    uint32_t    end,
    uint32_t    nodes,
    graph::Node *node,
    uint32_t    edges,
    graph::Edge *edge,
    uint32_t    connections,
    uint32_t    *connection,
    uint32_t    queues,
    uint32_t    *queue,
    uint32_t    edgeQueues,
    uint32_t    *edgeQueue)
{
    uint32_t c = begin + blockIdx.x * CUDA_THREADS + threadIdx.x;
    if(c >= end)
        return;

    Car car = _car[c];
    if(car.destination != UINT_MAX)
        return;

    car.destination = fast_rand(seed + threadIdx.x) % nodes;
    const graph::Node &from = node[car.destination];
    car.position        = vec2(from.x, from.y);
    if(nextEdge(seed, _car, c, car, nodes, node, edges, edge, connections, connection, queues, queue, edgeQueues, edgeQueue) < 2)
        car.destination = UINT_MAX;

    car.speed = 0.0f;
    _car[c] = car;
}

__global__ void SimulateCars(
    uint32_t    *seed,
    Car         *_prev,
    Car         *_car,
    uint32_t    begin,
    uint32_t    end,
    uint32_t    nodes,
    graph::Node *node,
    uint32_t    edges,
    graph::Edge *edge,
    uint32_t    connections,
    uint32_t    *connection,
    uint32_t    queues,
    uint32_t    *queue,
    uint32_t    edgeQueues,
    uint32_t    *edgeQueue)
{
    uint32_t c = begin + blockIdx.x * CUDA_THREADS + threadIdx.x;
    if(c >= end)
        return;

#ifdef CUDA_CACHE
    Car car = _car[c];
    Car prev = _prev[c];
#else
    __shared__ Car __car[CUDA_THREADS];
    __shared__ Car __prev[CUDA_THREADS];
    Car &car = __car[threadIdx.x] = _car[c];
    Car &prev = __prev[threadIdx.x] = _prev[c];
#endif
    if(car.destination == UINT_MAX)
        return;

    if(prev.destination != UINT_MAX)
    {
        const graph::Edge &edg = edge[prev.route];
        const float limit = edg.maxSpeed() ? edg.maxSpeed() * 5.0f / 18.0f / SIMULATION_FPS : 14.0f / SIMULATION_FPS;

        const graph::Node to = node[prev.destination];

        car.speed = min(limit, prev.speed + CAR_ACCELERATION / SIMULATION_FPS);
#ifdef CROSS_COLLISIONS
        if(vec2(to.x - car.position.x, to.y - car.position.y).length() < 4.0f * CAR_LENGTH)
            for(uint32_t e = to.edge, f = prev.destination < nodes - 1 ? node[prev.destination + 1].edge : connections; car.speed > 0.0f && e < f; ++ e)
                for(uint32_t q = edgeQueue[connection[e]], r = connection[e] < edges - 1 ? edgeQueue[connection[e] + 1] : queues; q < r; ++ q)
                {
                    uint32_t &d = queue[q];
                    if(d == c) continue;
                    Car &test = _prev[d];
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
            for(uint32_t q = edgeQueue[car.route], r = car.route < edges - 1 ? edgeQueue[car.route + 1] : queues; q < r; ++ q)
            {
                uint32_t &d = queue[q];
                if(d == c) continue;
                Car &test = _prev[d];
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

            if((before < now || now <= CAR_LENGTH / 4.0f) && !nextEdge(seed, _car, c, car, nodes, node, edges, edge, connections, connection, queues, queue, edgeQueues, edgeQueue)) // FINISHED
                car.destination = UINT_MAX;
        }

    }

    _car[c] = car;
}

/*****************************************************************************/
/************************* HELPERS *******************************************/

void cudaWorkerInitialize(graph::Graph &_graph)
{
    cuda.nodes = _graph.nodes;
    cudaError(cudaMalloc(&cuda.node, _graph.nodes * sizeof(graph::Node)));
    cudaError(cudaMemcpy(cuda.node, _graph.node, _graph.nodes * sizeof(graph::Node), cudaMemcpyHostToDevice));

    cuda.edges = _graph.edges;
    cudaError(cudaMalloc(&cuda.edge, _graph.edges * sizeof(graph::Edge)));
    cudaError(cudaMemcpy(cuda.edge, _graph.edge, _graph.edges * sizeof(graph::Edge), cudaMemcpyHostToDevice));

    cuda.connections = _graph.nodeEdges;
    cudaError(cudaMalloc(&cuda.connection, _graph.nodeEdges * sizeof(uint32_t)));
    cudaError(cudaMemcpy(cuda.connection, _graph.nodeEdge, _graph.nodeEdges * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void cudaWorkerCopyQueues(uint32_t *queue, uint32_t queueSize, uint32_t *edgeQueue, uint32_t edgeQueueSize)
{
    if(!cuda.queue)
    {
        cudaError(cudaMalloc(&cuda.queue, queueSize * sizeof(uint32_t)));
        cudaError(cudaMalloc(&cuda.edgeQueue, edgeQueueSize * sizeof(uint32_t)));
    }

    cudaError(cudaMemcpy(cuda.queue, queue, queueSize * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(cuda.edgeQueue, edgeQueue, edgeQueueSize * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cuda.queues = queueSize;
    cuda.edgeQueues = edgeQueueSize;
}

void cudaWorkerCopyCars(Car *car, uint32_t count, uint32_t begin, uint32_t end)
{
    if(!cuda.car)
    {
        cudaError(cudaMalloc(&cuda.car, count * sizeof(Car)));
        cudaError(cudaMalloc(&cuda.prev, count * sizeof(Car)));
        cudaError(cudaMalloc(&cuda.seed, CUDA_THREADS * sizeof(uint32_t)));
        uint32_t *seed = new uint32_t[CUDA_THREADS];
        for(uint32_t c = 0; c < CUDA_THREADS; ++ c)
            seed[c] = 1LLU * UINT_MAX * count * c / CUDA_THREADS;

        cudaError(cudaMemcpy(cuda.seed, seed, CUDA_THREADS * sizeof(uint32_t), cudaMemcpyHostToDevice));
        delete[] seed;
    }

    cuda.cars = count;
    cudaError(cudaMemcpy(cuda.car, car, count * sizeof(Car), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(cuda.prev, cuda.car, count * sizeof(Car), cudaMemcpyDeviceToDevice));

    cuda.begin      = begin;
    cuda.end        = end;
    cuda.blocks     = max(1, (cuda.end - cuda.begin) / CUDA_THREADS);
}

void cudaWorkerReplenishCars(void)
{
    ReplenishCars<<<cuda.blocks, CUDA_THREADS>>>(cuda.seed, cuda.car, cuda.begin, cuda.end, cuda.nodes, cuda.node, cuda.edges, cuda.edge, cuda.connections, cuda.connection, cuda.queues, cuda.queue, cuda.edgeQueues, cuda.edgeQueue);
    cudaError(cudaPeekAtLastError());
    cudaError(cudaDeviceSynchronize());
}

/*
void cudaWorkerRoute(void)
{
}*/

void cudaWorkerSimulate(void)
{
#ifndef __CPU__
    cudaError(cudaMemcpy(cuda.prev, cuda.car, (cuda.end - cuda.begin) * sizeof(Car), cudaMemcpyDeviceToDevice));
#endif // __CPU__

    SimulateCars<<<cuda.blocks, CUDA_THREADS>>>(cuda.seed, cuda.prev, cuda.car, cuda.begin, cuda.end, cuda.nodes, cuda.node, cuda.edges, cuda.edge, cuda.connections, cuda.connection, cuda.queues, cuda.queue, cuda.edgeQueues, cuda.edgeQueue);
    cudaError(cudaPeekAtLastError());
    cudaError(cudaDeviceSynchronize());
}

void cudaWorkerUpdate(Car *begin, uint32_t count)
{
    cudaError(cudaMemcpy(begin, cuda.car, count * sizeof(Car), cudaMemcpyDeviceToHost));
}

void cudaWorkerFinished(void)
{
    cudaFree(cuda.car);
    cudaFree(cuda.prev);
    cudaFree(cuda.node);
    cudaFree(cuda.edge);
    cudaFree(cuda.connection);
}
