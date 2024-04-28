import QtQuick
import QtQuick.Controls.Material

Rectangle {
    id: headerRoot
    height: 50
    width: parent.width
    anchors{
        left: parent.left
        top: parent.top
        topMargin: 15
    }
    color: "transparent"

    TextTitle{
        id: headerTitle
        width: parent.width
        text: "Agent Diagnostics Dashboard"
    }
}