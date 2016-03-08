#include "defines.h"
#include "consoleWidget.h"

using namespace trafficsim;
using namespace trafficsim::gui;

ConsoleWidget::ConsoleWidget(QWidget *_parent/* = nullptr*/)
:QPlainTextEdit(_parent)
,ps("/> ")
,fixedPosition(3)
{
    // Set colors
    QPalette colors = palette();
    colors.setColor(QPalette::Base, QColor(0,    0,      0));
    colors.setColor(QPalette::Text, QColor(255,  255,    255));

    setPalette(colors);

    // Disable unused stuff
    setUndoRedoEnabled(false);

    // Initial text
    insertPlainText(ps);

    connect(this, SIGNAL(cursorPositionChanged()), this, SLOT(cursorPositionChanged()));
}

void ConsoleWidget::cursorPositionChanged(void)
{
    QTextCursor _cursor = textCursor();
    if(_cursor.position() < fixedPosition)
    {
        _cursor.setPosition(fixedPosition);
        setTextCursor(_cursor);
    }
}

void ConsoleWidget::keyPressEvent(QKeyEvent *_event)
{
    QTextCursor _cursor = textCursor();
    bool passthrough = fixedPosition <= _cursor.position();
    switch(_event->key())
    {
        case Qt::Key_Backspace:
            passthrough = fixedPosition < _cursor.position();
            break;

        case Qt::Key_Return:
            {
                passthrough = false;
                // move cursor to end of line
                _cursor.movePosition(QTextCursor::End);
                setTextCursor(_cursor);

                // get command and passthrough to sb
                QString cmd = toPlainText().right(_cursor.position() - fixedPosition);

                insertPlainText("\n");
                // PASSTHROUGH: TODO
                QString result = "error: \"" + cmd + "\"\n";

                // insert result
                insertPlainText(result);

                // insert PS
                insertPlainText(ps);
                fixedPosition = _cursor.position();
                ensureCursorVisible();
            }

            break;

        case Qt::Key_Tab:
            passthrough = false;
            // handle autocompletion
            break;

        case Qt::Key_PageUp:
        case Qt::Key_PageDown:
            passthrough = false;
            // handle history
            break;
    }

    if(passthrough)
        QPlainTextEdit::keyPressEvent(_event);
}
