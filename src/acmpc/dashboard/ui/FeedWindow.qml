import QtQuick
import QtQuick.Controls.Material

 Image {
        id: feedImageRoot
        anchors{
            top: parent.top
            topMargin: 5
            left: parent.left
            leftMargin: 5
            right: parent.right
            rightMargin: 5
            
        }
        fillMode: Image.PreserveAspectFit
        property bool counter: false
        property string source_root: ""
        cache: false

        function reloadImage() {
            counter = !counter
            source = source_root + "?id=" + counter
        }
        Component.onCompleted: {
            reloadImage()
        }
}