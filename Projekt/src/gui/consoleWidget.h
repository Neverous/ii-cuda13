#ifndef __CONSOLE_WIDGET_H__
#define __CONSOLE_WIDGET_H__

#include <QPlainTextEdit>
#include <QKeyEvent>

namespace trafficsim
{

namespace gui
{

class ConsoleWidget: public QPlainTextEdit
{
    Q_OBJECT

    QString ps;
    int fixedPosition;

    public:
        ConsoleWidget(QWidget *_parent = nullptr);

    public slots:
        void cursorPositionChanged(void);

    protected:
        void keyPressEvent(QKeyEvent *_event);
}; // class ConsoleWidget

} // namespace gui

} // namespace trafficsim

#endif // __CONSOLE_WIDGET_H__
