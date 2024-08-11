import QtQuick
import QtQuick.Controls.Material


Item{
    anchors.fill: parent
    Header{
        id: header
    }
    FeedGrid{
        id: feedGrid
        anchors{
            top: header.bottom
            bottom: parent.bottom
        }
    }
}