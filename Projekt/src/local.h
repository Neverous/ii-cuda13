#ifndef __LOCAL_H__
#define __LOCAL_H__

#include <vector>
#include "graph/graph.h"
#include "trafficsim/car.h"

namespace trafficsim
{

using namespace std;

struct Local
{
    graph::Graph        graph;
    vector<Car>         car;
    vector<Car>         prev;

    vector<uint32_t>    edgeQueue;
    vector<uint32_t>    queue;
}; // struct Local

} // namespace trafficsim

#endif // __LOCAL_H__
