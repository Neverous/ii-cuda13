#ifndef __TRAFFICSIM_WINDOW_H__
#define __TRAFFICSIM_WINDOW_H__

#include <QWidget>
#include <QVBoxLayout>
#include <QTabWidget>

#include "drawerWidget.h"
#include "consoleWidget.h"

namespace trafficsim
{

namespace gui
{

class TrafficSimWindow: public QWidget
{
    Q_OBJECT

    QVBoxLayout     layout;
    DrawerWidget    opengl;
//    QTabWidget      tabs;

//    ConsoleWidget   console;
//    QWidget         preferences;

    public:
        TrafficSimWindow(Local &_local, QWidget *_parent = nullptr);
}; // class TraffixSimWindow

} // namespace gui

} // namespace trafficsim

#endif // __TRAFFICSIM_WINDOW_H__
