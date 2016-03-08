#ifndef __IMPORT_H__
#define __IMPORT_H__

#include <limits>
#include <algorithm>
#include <bitset>
#include <set>
#include <vector>
#include <QDebug>

#include "OSMPBF/OSMPBFReader.h"
#include "projection/mercator.h"
#include "graph/graph.h"

namespace trafficsim
{

namespace import
{

using namespace std;
using namespace trafficsim::projection;

class Statistics;
template<size_t bits> class Filter;
template<size_t bits> class Generator;

static const set<string> validHighway = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "living_street",
    "residential",
    "unclassified",
    "track",
    "road",
};

template<size_t bits>
class Filter
{
    friend class Generator<bits>;

    struct NodeInfo
    {
        uint64_t    count;
        uint64_t    min;
        uint64_t    max;
    } node;

    struct WayInfo
    {
        uint64_t    count;
        uint64_t    min;
        uint64_t    max;
    } way;

    bitset<bits>    *filter;
    bitset<bits>    *avail;
    uint32_t        *mapping;
    uint32_t        count;
    public:
        Filter(void);
        ~Filter(void);

        void handleNode(uint64_t id, float lon, float lat, Tags &tags);
        void handleWay(uint64_t id, Tags &tags, const Nodes &nodes);
        void handleRelation(uint64_t id, const Tags &tags, const References &refs);

        void finish(void);
        uint32_t get(uint32_t id);
}; // class Filter

template<size_t bits>
class Generator
{
    public:
        graph::Graph    graph;

    private:
        Filter<bits>    &filter;
        vector<graph::Node> nodes;
        vector<graph::Edge> edges;
        vector<uint32_t>    nodeEdges;
        float minX;
        float minY;

    public:
        Generator(Filter<bits> &_filter);

        void handleNode(uint64_t id, float lon, float lat, Tags &tags);
        void handleWay(uint64_t id, Tags &tags, const Nodes &nodes);
        void handleRelation(uint64_t id, const Tags &tags, const References &refs);

        void finish(void);
        void clean(void);
}; // class Generator

template<size_t bits>
inline
Filter<bits>::Filter(void)
:node()
,way()
,filter(nullptr)
,avail(nullptr)
,mapping(nullptr)
,count(0)
{
    filter = new bitset<bits>();
    avail = new bitset<bits>();

    node.min = numeric_limits<uint64_t>::max();
    node.max = numeric_limits<uint64_t>::min();

    way.min = numeric_limits<uint64_t>::max();
    way.max = numeric_limits<uint64_t>::min();
}

template<size_t bits>
inline
Filter<bits>::~Filter(void)
{
    if(filter)
        delete filter;

    if(mapping)
        delete[] mapping;

    filter = nullptr;
    mapping = nullptr;
}

template<size_t bits>
inline
void Filter<bits>::handleNode(uint64_t id, float/* lon*/, float/* lat*/, Tags &/*tags*/)
{
    ++ node.count;
    node.min = min(node.min, id);
    node.max = max(node.max, id);
    avail->set(id);
}

template<size_t bits>
inline
void Filter<bits>::handleWay(uint64_t id, Tags &tags, const Nodes &nodes)
{
    if(!tags.count("highway") || !validHighway.count(tags["highway"]))
        return;

    assert(nodes.size() > 1);
    ++ way.count;
    way.min = min(way.min, id);
    way.max = max(way.max, id);

    for(const auto &node: nodes)
        filter->set(node);
}

template<size_t bits>
inline
void Filter<bits>::handleRelation(uint64_t/* id*/, const Tags &/*tags*/, const References &/*refs*/)
{
    throw runtime_error("Not supposed to run");
}

template<size_t bits>
inline
void Filter<bits>::finish(void)
{
    qDebug() << "Read" << node.count << "nodes ( from" << node.min << "to" << node.max << ")";
    qDebug() << "Read" << way.count << "ways ( from" << way.min << "to" << way.max << ")";

    *filter &= *avail;
    count = filter->count();
    qDebug() << "Left" << count << "points.";
    mapping = new uint32_t[count];
    uint32_t m = 0;
    for(uint64_t f = node.min; f <= node.max; ++ f)
        if(filter->test(f))
            mapping[m ++] = f;

    assert(m == count);
    delete filter;
    delete avail;
    filter = nullptr;
    avail = nullptr;
}

template<size_t bits>
inline
uint32_t Filter<bits>::get(uint32_t id)
{
    uint32_t *pos = lower_bound(mapping, mapping + count, id);
    if(pos == mapping + count || *pos != id)
        return 0;

    return 1 + pos - mapping;
}

template<size_t bits>
inline
Generator<bits>::Generator(Filter<bits> &_filter)
:graph()
,filter(_filter)
,nodes()
,edges()
,nodeEdges()
,minX(numeric_limits<float>::max())
,minY(numeric_limits<float>::max())
{
    nodes.resize(filter.count);
}

template<size_t bits>
inline
void Generator<bits>::handleNode(uint64_t _id, float lon, float lat, Tags &/*tags*/)
{
    uint32_t id = filter.get(_id);
    if(id == 0 || filter.mapping[id - 1] != _id)
        return; // Not found

    assert(nodes[id - 1].x == 0.0f && nodes[id - 1].y == 0.0f);
    nodes[id - 1] = graph::Node(mercator::lonToMet(lon), mercator::latToMet(lat));
    minX = min(minX, nodes[id - 1].x);
    minY = min(minY, nodes[id - 1].y);
}

template<size_t bits>
inline
void Generator<bits>::handleWay(uint64_t/* id*/, Tags &tags, const Nodes &_nodes)
{
    if(!tags.count("highway") || !validHighway.count(tags["highway"]))
        return;

    graph::Edge edge(filter.get(_nodes[0]), 0, tags["oneway"] == "yes" || tags["junction"] == "roundabout" || tags["junction"] == "mini_roundabout" || tags["highway"] == "motorway");
    edge.setLanes(max(1, stoi("0" + tags["lanes"]) / (1 + !edge.oneway())));
    edge.setMaxSpeed(stoi("0" + tags["maxspeed"]));
    assert(!edge.from || _nodes[0] == filter.mapping[edge.from - 1]);
    for(uint32_t n = 1; n < _nodes.size(); ++ n)
    {
        edge.to = filter.get(_nodes[n]);
        assert(!edge.to || _nodes[n] == filter.mapping[edge.to - 1]);
        if(edge.from > 0 && edge.to > 0)
        {
            -- edge.from;
            -- edge.to;
            edges.push_back(edge);

            ++ edge.from;
            ++ edge.to;
        }

        edge.from = edge.to;
    }
}

template<size_t bits>
inline
void Generator<bits>::handleRelation(uint64_t/* id*/, const Tags &/*tags*/, const References &/*refs*/)
{
    throw runtime_error("Not supposed to run");
}

template<size_t bits>
inline
void Generator<bits>::finish(void)
{
    nodeEdges.resize(edges.size() * 2);
    vector<uint32_t> weight(nodes.size());
    for(const auto &edge: edges)
    {
        ++ weight[edge.from];
        ++ weight[edge.to];
    }

    nodes[0].edge = 0;
    nodes[0].x -= minX;
    nodes[0].y -= minY;
    for(uint32_t n = 1; n < nodes.size(); ++ n)
    {
        nodes[n].edge = nodes[n - 1].edge + weight[n - 1];
        nodes[n].x -= minX;
        nodes[n].y -= minY;
    }

    for(uint32_t e = 0; e < edges.size(); ++ e)
    {
        nodeEdges[nodes[edges[e].from].edge ++] = e;
        nodeEdges[nodes[edges[e].to].edge ++] = e;
    }

    nodes[0].edge = 0;
    for(uint32_t n = 1; n < nodes.size(); ++ n)
        nodes[n].edge = nodes[n - 1].edge + weight[n - 1];

    graph.nodes       = nodes.size();
    graph.edges       = edges.size();
    graph.nodeEdges   = nodeEdges.size();

    graph.node        = &nodes[0];
    graph.edge        = &edges[0];
    graph.nodeEdge    = &nodeEdges[0];
    qDebug() << "Read" << graph.nodes << "nodes";
    qDebug() << "Read" << graph.edges << "edges";
}

template<size_t bits>
inline
void Generator<bits>::clean(void)
{
    graph.node      = nullptr;
    graph.edge      = nullptr;
    graph.nodeEdge  = nullptr;
}

} // namespace import

} // namespace trafficsim

#endif // __IMPORT_H__

