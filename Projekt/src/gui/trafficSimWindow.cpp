#include "defines.h"
#include "trafficSimWindow.h"

using namespace trafficsim;
using namespace trafficsim::gui;

TrafficSimWindow::TrafficSimWindow(Local &_local, QWidget *_parent/* = nullptr*/)
:QWidget(_parent)
,layout(this)
,opengl(_local, this)
//,tabs(this)
//,console()
//,preferences()
{
    setWindowTitle("TrafficSim v" VERSION_FULL);

    // default window size
    resize(800, 600);

    // insides
    layout.setSpacing(0);
    layout.setContentsMargins(0, 0, 0, 0);

    opengl.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    opengl.setCursor(Qt::CrossCursor);
    //tabs.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    //console.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    //preferences.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    //tabs.addTab(&console, tr("Console"));
    //tabs.addTab(&preferences, tr("Preferences"));

    layout.addWidget(&opengl, 3);
    //layout.addWidget(&tabs, 1);
}
