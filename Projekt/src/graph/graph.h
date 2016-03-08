#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

#ifndef __cuda__
    #define __cuda__
#endif // CUDA

namespace trafficsim
{

namespace graph
{

using namespace std;

struct Node;
struct Edge;

const uint8_t   VERSION = 3;
const float     LANE_WIDTH = 3.0f;

struct Node
{
    float      x;
    float      y;
    uint32_t    edge;

    Node(float _x = 0.0f, float _y = 0.0f, uint32_t _edge = 0);
}; // struct Node [20]

#define ONEWAY_MASK     0x8000
#define LANES_MASK      0x7800
#define MAXSPEED_MASK   0x07FF
struct Edge
{
    uint32_t    from;
    uint32_t    to;

    // [ONEWAY][LANES][MAXSPEED]
    uint16_t     blob;

    Edge(uint32_t _from = 0, uint32_t _to = 0, bool _oneway = false, uint8_t _lanes = 0, uint16_t _maxSpeed = 0);

    void setOneway(bool _oneway);
    void setLanes(uint8_t _lanes);
    void setMaxSpeed(uint16_t _maxSpeed);

    __cuda__ bool oneway(void) const;
    __cuda__ uint8_t lanes(void) const;
    __cuda__ uint16_t maxSpeed(void) const;
}; // struct Edge [10]

struct Graph
{
    uint8_t     version;

    uint32_t    nodes;
    Node        *node;

    uint32_t    edges;
    Edge        *edge;

    uint32_t    nodeEdges;
    uint32_t    *nodeEdge;

    Graph(void);
    ~Graph(void);

    size_t save(const char *filename);
    void load(const char *filename);
}; // class Graph

inline
Node::Node(float _x/* = 0.0f*/, float _y/* = 0.0f*/, uint32_t _edge/* = 0*/)
:x(_x)
,y(_y)
,edge(_edge)
{
}

inline
Edge::Edge(uint32_t _from/* = 0*/, uint32_t _to/* = 0*/, bool _oneway/* = false*/, uint8_t _lanes/* = 0*/, uint16_t _maxSpeed/* = 0*/)
:from(_from)
,to(_to)
,blob(0)
{
    setOneway(_oneway);
    setLanes(_lanes);
    setMaxSpeed(_maxSpeed);
}

inline
void Edge::setOneway(bool _oneway)
{
    blob = (blob & ~ONEWAY_MASK) + ((uint16_t) _oneway << 15);

}

inline
void Edge::setLanes(uint8_t _lanes)
{
    blob = (blob & ~LANES_MASK) + ((uint16_t) _lanes << 11);
}

inline
void Edge::setMaxSpeed(uint16_t _maxSpeed)
{
    blob = (blob & ~MAXSPEED_MASK) + _maxSpeed;
}

inline
__cuda__ bool Edge::oneway(void) const
{
    return (blob & ONEWAY_MASK) >> 15;
}

inline
__cuda__ uint8_t Edge::lanes(void) const
{
    return (blob & LANES_MASK) >> 11;
}

inline
__cuda__ uint16_t Edge::maxSpeed(void) const
{
    return (blob & MAXSPEED_MASK);
}

inline
Graph::Graph(void)
:nodes(0)
,node(NULL)
,edges(0)
,edge(NULL)
,nodeEdges(0)
,nodeEdge(NULL)
{
}

inline
Graph::~Graph(void)
{
    if(node)
        delete[] node;

    if(edge)
        delete[] edge;

    if(nodeEdge)
        delete[] nodeEdge;

    node = NULL;
    edge = NULL;
    nodeEdge = NULL;
}

inline
size_t Graph::save(const char *filename)
{
    version = VERSION;
    int file = open(filename, O_WRONLY);
    if(!file)
        throw runtime_error("Couldn't open file");

    size_t bytes = 0;
    bytes += write(file, &version, sizeof(uint8_t));
    bytes += write(file, &nodes, sizeof(uint32_t));
    bytes += write(file, node, sizeof(Node) * nodes);
    bytes += write(file, &edges, sizeof(uint32_t));
    bytes += write(file, edge, sizeof(Edge) * edges);
    bytes += write(file, &nodeEdges, sizeof(uint32_t));
    bytes += write(file, nodeEdge, sizeof(uint32_t) * nodeEdges);

    close(file);
    return bytes;
}

inline
void Graph::load(const char *filename)
{
    int file = open(filename, O_RDONLY);
    if(!file)
        throw runtime_error("Couldn't open file");

    read(file, &version, sizeof(uint8_t));
    if(version != VERSION)
        throw runtime_error("Invalid archive version");

    read(file, &nodes, sizeof(uint32_t));
    if(!(node = new Node[nodes]))
        throw runtime_error("Insufficient memory");

    read(file, node, sizeof(Node) * nodes);
    read(file, &edges, sizeof(uint32_t));
    if(!(edge = new Edge[edges]))
        throw runtime_error("Insufficient memory");

    read(file, edge, sizeof(Edge) * edges);
    read(file, &nodeEdges, sizeof(uint32_t));
    if(!(nodeEdge = new uint32_t[nodeEdges]))
        throw runtime_error("Insufficient memory");

    read(file, nodeEdge, sizeof(uint32_t) * nodeEdges);

    close(file);
}

} // namespace graph

} // namespace trafficsim

#endif // __GRAPH_H__
