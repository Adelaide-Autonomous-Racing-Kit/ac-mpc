import QtQuick
import QtQuick.Controls.Material

Rectangle {
        id: sessionSummaryRoot
        anchors{
            top: parent.top
            topMargin: 5
            left: parent.left
            leftMargin: 5
            right: parent.right
            rightMargin: 5
            bottom: parent.bottom
            bottomMargin: 5
        }

        Text {
            text: sessionInfo.laptime
            font.pixelSize: 48
            color: "black"
        }

}