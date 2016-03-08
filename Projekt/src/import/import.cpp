#include "import.h"

#include <cstdio>
#include <chrono>

#include "OSMPBF/OSMPBFReader.h"

using namespace std;
using namespace trafficsim;
using namespace trafficsim::import;

void qMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);

typedef Filter<4294967296U> NodeFilter;
NodeFilter filter;

int main(int argc, char *argv[])
{
    qInstallMessageHandler(qMessageHandler);
    if(argc != 3)
    {
        fprintf(stderr, "usage: %s mapfile.osm.pbf output.dat\n", argv[0]);
        return 1;
    }

    qDebug() << "Started import";

    qDebug() << "Filtering nodes";
    OSMPBFReader<NodeFilter> reader2(argv[1], filter, true, true, false);
    reader2.read();
    filter.finish();

    qDebug() << "Generating map";
    Generator<4294967296U> generator(filter);
    OSMPBFReader<Generator<4294967296U> > reader3(argv[1], generator, true, true, false);
    reader3.read();
    generator.finish();

    qDebug() << "Saving graph";
    size_t bytes = generator.graph.save(argv[2]);

    qDebug() << "Saved" << bytes / 1024 << "KB";
    generator.clean();

    return 0;
}

void qMessageHandler(QtMsgType type, const QMessageLogContext &/*context*/, const QString &msg)
{
    using namespace chrono;

    char            output[1024]    = {};
    const char      *level          = nullptr;
    auto            now             = system_clock::now();
    auto            now_c           = system_clock::to_time_t(now);
    unsigned int    millisec        = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    QByteArray      local           = msg.toLocal8Bit();
    int             o               = 0;

    switch(type)
    {
        case QtDebugMsg:
            level = "DEBUG";
            break;

        case QtWarningMsg:
            level = "WARNING";
            break;

        case QtCriticalMsg:
            level = "CRITICAL";
            break;

        case QtFatalMsg:
            level = "FATAL";
            break;
    }

    o = strftime(output, 1023, "[%Y-%m-%d %H:%M:%S", localtime(&now_c));
    if(o > 1023) o = 1024;
    o += snprintf(output + o, 1023 - o, ":%03d] %s: %s\n", millisec, level, local.constData());
    fputs(output, stderr);
    fflush(stderr);

    if(type == QtFatalMsg)
        abort();
}
