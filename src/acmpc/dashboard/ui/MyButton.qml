import QtQuick
import QtQuick.Controls.Material

Button{
    id: root

    contentItem: TextSubtitle {
        anchors.fill: parent
        text: root.text
    }
    background: Rectangle {
        opacity: root.hovered ? 0.4 : 0.1
        radius: 10
    }
}