#include "defines.h"
#include <QApplication>
#include <chrono>
#include <mutex>

#include "local.h"
#include "trafficsim/simulation.h"
#include "gui/trafficSimWindow.h"

using namespace std;
using namespace trafficsim;

void qMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);

int main(int argc, char *argv[])
{
    Local local;

    local.car.resize(CARS);
    local.prev.resize(CARS);
    for(auto &car: local.car)
        car.destination = numeric_limits<uint32_t>::max();

    qInstallMessageHandler(qMessageHandler);
    QApplication app(argc, argv);

    local.graph.load("map.dat");
    gui::TrafficSimWindow window(local);
    window.show();

    Simulation manager(local);
    manager.start();

    int ret = app.exec();
    manager.stop();
    qDebug() << "EVERYTHING SHOULD BE STOPPED BY NOW";
    return ret;
}

void qMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    static mutex lock;
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
    o += snprintf(output + o, 1023 - o, ":%03d] %s: (%s:%d) %s\n", millisec, level, context.function, context.line, local.constData());
    {
        std::lock_guard<std::mutex> _lock(lock);
        fputs(output, stderr);
        fflush(stderr);
    }

    if(type == QtFatalMsg)
        abort();
}
