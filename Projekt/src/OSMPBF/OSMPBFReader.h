#ifndef __OSMPBF_READER_H__
#define __OSMPBF_READER_H__

#include <cstdint>
#include <netinet/in.h>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <zlib.h>

#include <QDebug>

#include "fileformat.pb.h"
#include "osmformat.pb.h"

using namespace std;
using namespace OSMPBF;

typedef unordered_map<string, string> Tags;
typedef vector<uint64_t>    Nodes;
struct Reference
{
    Relation::MemberType    memberType;
    uint64_t                memberID;
    string                  role;

    Reference(void);
    Reference(const Relation::MemberType &_memberType, uint64_t _memberID, const string &_role);
}; // struct Reference

typedef vector<Reference> References;

template<class Actor>
class OSMPBFReader
{
    Actor       &actor;
    ifstream    input;

    bool        nodes;
    bool        ways;
    bool        relations;

    public:
        OSMPBFReader(const string &filename, Actor &_actor, bool _nodes = true, bool _ways = true, bool _relations = true);
        ~OSMPBFReader(void);

        void read(void);

    protected:
        template<class Array>
        bool getTags(const Array &array, const PrimitiveBlock &block, Tags &tags);

        bool readHeader(BlobHeader &header);

        bool readBlob(const BlobHeader &header, string &blob);
        bool readBlobRaw(const Blob &_blob, string &blob);
        bool readBlobZlib(const Blob &_blob, string &blob);

        bool parsePrimitiveBlock(const string &blob);
        bool parseNodes(const PrimitiveGroup &group, const PrimitiveBlock &block);
        bool parseDenseNodes(const PrimitiveGroup &group, const PrimitiveBlock &block);
        bool parseWays(const PrimitiveGroup &group, const PrimitiveBlock &block);
        bool parseRelations(const PrimitiveGroup &group, const PrimitiveBlock &block);

    private:
        bool read(string &out, int32_t bytes);
}; // class OSMPBFReader

inline
Reference::Reference(void)
:memberType()
,memberID(0)
,role()
{
}

inline
Reference::Reference(const Relation::MemberType &_memberType, uint64_t _memberID, const string &_role)
:memberType(_memberType)
,memberID(_memberID)
,role(_role)
{
}

template<class Actor>
inline
OSMPBFReader<Actor>::OSMPBFReader(const string &filename, Actor &_actor, bool _nodes/* = true*/, bool _ways/* = true*/, bool _relations/* = true*/)
:actor(_actor)
,input(filename.c_str(), ios::binary)
,nodes(_nodes)
,ways(_ways)
,relations(_relations)
{
    if(!input.is_open())
        throw runtime_error("Unable to open file!");
}

template<class Actor>
inline
OSMPBFReader<Actor>::~OSMPBFReader(void)
{
    google::protobuf::ShutdownProtobufLibrary();
}

template<class Actor>
inline
void OSMPBFReader<Actor>::read(void)
{
    BlobHeader  header;
    string      blob;
    while(!input.eof() && readHeader(header))
    {
        if(!readBlob(header, blob))
            throw runtime_error("Couldn't read blob!");

        if(header.type() == "OSMData")
            parsePrimitiveBlock(blob);

        else if(header.type() != "OSMHeader")
            throw runtime_error("Unknown blob type!");
    }
}

template<class Actor>
template<class Array>
inline
bool OSMPBFReader<Actor>::getTags(const Array &array, const PrimitiveBlock &block, Tags &tags)
{
    tags.clear();
    for(int a = 0; a < array.keys_size(); ++ a)
    {
        uint64_t k = array.keys(a);
        uint64_t v = array.vals(a);
        const string &key = block.stringtable().s(k);
        const string &val = block.stringtable().s(v);

        tags[key] = val;
    }

    return true;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::readHeader(BlobHeader &header)
{
    int32_t size = 0;
    string  data;
    if(!input.read((char *) &size, 4))
        return false;

    size = ntohl(size);
    if(!read(data, size))
        throw runtime_error("Unable to read header!");

    if(!header.ParseFromArray(data.c_str(), size))
        throw runtime_error("Unable to parse header!");

    return true;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::readBlob(const BlobHeader &header, string &blob)
{
    int32_t size = header.datasize();
    Blob    _blob;
    if(!read(blob, size))
        throw runtime_error("Unable to read blob!");

    if(!_blob.ParseFromArray(blob.c_str(), size))
        throw runtime_error("Unable to parse blob!");

    if(_blob.has_raw())
        return readBlobRaw(_blob, blob);

    if(_blob.has_zlib_data())
        return readBlobZlib(_blob, blob);

    return false;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::readBlobRaw(const Blob &_blob, string &blob)
{
    int32_t size = _blob.raw().size();
    if(size != _blob.raw_size())
        throw runtime_error("Blob size mismatch!");

    blob.resize(size);
    return true;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::readBlobZlib(const Blob &_blob, string &blob)
{
    int32_t size = _blob.zlib_data().size();
    blob.resize(_blob.raw_size());

    z_stream zlib;
    zlib.next_in    = (unsigned char *) _blob.zlib_data().c_str();
    zlib.avail_in   = size;
    zlib.next_out   = (unsigned char *) &blob[0];
    zlib.avail_out  = _blob.raw_size();
    zlib.zalloc     = Z_NULL;
    zlib.zfree      = Z_NULL;
    zlib.opaque     = Z_NULL;

    if(inflateInit(&zlib) != Z_OK)
        throw runtime_error("Failed to initialize zlib stream!");

    if(inflate(&zlib, Z_FINISH) != Z_STREAM_END)
        throw runtime_error("Failed to inflate zlib stream!");

    if(inflateEnd(&zlib) != Z_OK)
        throw runtime_error("Failed to deinit zlib stream!");

    return true;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::parsePrimitiveBlock(const string &blob)
{
    PrimitiveBlock  block;
    if(!block.ParseFromArray(blob.c_str(), blob.size()))
        throw runtime_error("Unable to parse primitive block!");

    for(int g = 0, groups = block.primitivegroup_size(); g < groups; ++ g)
    {
        PrimitiveGroup  group = block.primitivegroup(g);
        if(nodes && !parseNodes(group, block))
            return false;

        if(ways && !parseWays(group, block))
            return false;

        if(relations && !parseRelations(group, block))
            return false;
    }

    return true;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::parseNodes(const PrimitiveGroup &group, const PrimitiveBlock &block)
{
    for(int n = 0; n < group.nodes_size(); ++ n)
    {
        Node node = group.nodes(n);
        uint64_t lon = block.lon_offset() + block.granularity() * node.lon();
        uint64_t lat = block.lat_offset() + block.granularity() * node.lat();
        Tags tags;
        if(!getTags(node, block, tags))
            return false;

        actor.handleNode(node.id(), lon * 0.000000001f, lat * 0.000000001f, tags);
    }

    if(group.has_dense() && !parseDenseNodes(group, block))
        return false;

    return true;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::parseDenseNodes(const PrimitiveGroup &group, const PrimitiveBlock &block)
{
    DenseNodes  dense   = group.dense();
    uint64_t    id      = 0;
    uint64_t    lon     = 0;
    uint64_t    lat     = 0;
    int         current = 0;

    for(int d = 0; d < dense.id_size(); ++ d)
    {
        id += dense.id(d);
        lon += block.lon_offset() + block.granularity() * dense.lon(d);
        lat += block.lat_offset() + block.granularity() * dense.lat(d);

        Tags tags;
        while(current < dense.keys_vals_size() && dense.keys_vals(current) != 0)
        {
            uint64_t k = dense.keys_vals(current);
            uint64_t v = dense.keys_vals(current + 1);
            current += 2;

            const string &key = block.stringtable().s(k);
            const string &val = block.stringtable().s(v);
            tags[key] = val;
        }

        ++ current;
        actor.handleNode(id, lon * 0.000000001f, lat * 0.000000001f, tags);
    }

    return true;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::parseWays(const PrimitiveGroup &group, const PrimitiveBlock &block)
{
    for(int w = 0; w < group.ways_size(); ++ w)
    {
        Way         way = group.ways(w);
        uint64_t    node = 0;
        Nodes       nodes;

        for(int n = 0; n < way.refs_size(); ++ n)
        {
            node += way.refs(n);
            nodes.push_back(node);
        }

        Tags tags;
        if(!getTags(way, block, tags))
            return false;

        actor.handleWay(way.id(), tags, nodes);
    }

    return true;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::parseRelations(const PrimitiveGroup &group, const PrimitiveBlock &block)
{
    for(int r = 0; r < group.relations_size(); ++ r)
    {
        Relation    relation = group.relations(r);
        uint64_t    id = 0;
        References  refs;
        for(int m = 0; m < relation.memids_size(); ++ m)
        {
            id += relation.memids(m);
            refs.push_back(Reference(relation.types(m), id, block.stringtable().s(relation.roles_sid(m))));
        }

        Tags tags;
        if(!getTags(relation, block, tags))
            return false;

        actor.handleRelation(relation.id(), tags, refs);
    }

    return true;
}

template<class Actor>
inline
bool OSMPBFReader<Actor>::read(string &out, int32_t bytes)
{
    out.resize(bytes);
    return input.read(&out[0], bytes);
}

#endif // __OSMPBF_READER_H__
