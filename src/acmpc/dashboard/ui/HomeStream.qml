import QtQuick
import QtQuick.Controls.Material


Item{
    anchors.fill: parent
    FeedGridStream{
        id: feedGrid
        anchors{
            top: parent.top
            bottom: parent.bottom
        }
    }
}