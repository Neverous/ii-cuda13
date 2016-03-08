#ifndef __PROFILER_H__
#define __PROFILER_H__

#include <cstdint>
#include <cstdio>
#include <unistd.h>
#include <chrono>

#define PGUARD(Name) TGUARD(Name);

#define TGUARD(Name) TimeGuard __tg(Name);

using namespace std::chrono;

class TimeGuard
{
    const char                  *name;
    system_clock::time_point    start;

    public:
        TimeGuard(const char *_name);
        ~TimeGuard(void);

    private:
        double calcUnit(size_t value, const char *&unit) const;
}; // class TimeGuard

// TIMEGUARD
inline
TimeGuard::TimeGuard(const char *_name)
:name(_name)
,start(system_clock::now())
{
    fprintf(stderr, "TimeGuard::%s started\n", _name);
}

inline
TimeGuard::~TimeGuard(void)
{
    const char *unit = nullptr;
    double value = calcUnit(duration_cast<microseconds>(system_clock::now() - start).count(), unit);
    fprintf(stderr, "TimeGuard::%s took %0.5lf%s\n", name, value, unit);
}

inline
double TimeGuard::calcUnit(size_t _value, const char *&unit) const
{
    double value = _value;
#define TIME_POINT(Limit, Unit) \
    if(value < Limit)           \
    {                           \
        unit = Unit;            \
        return value;           \
    }                           \
                                \
    value /= Limit;

    TIME_POINT(1000,    "us");
    TIME_POINT(1000,    "ms");
    TIME_POINT(60,      "s");
    TIME_POINT(60,      "m");
    TIME_POINT(24,      "h");
    unit = "d";
    return value;
#undef TIME_POINT
}

#endif // __PROFILER_H__
