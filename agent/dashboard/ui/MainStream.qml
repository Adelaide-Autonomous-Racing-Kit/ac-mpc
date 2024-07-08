import QtQuick
import QtQuick.Controls.Material

ApplicationWindow {
    title: "Agent Dashboard"
    width: 1920
    height: 1080
    visible: true
    Material.theme: Material.Dark


    Loader {
        id: mainLoader
        anchors.fill: parent
        source: "HomeStream.qml"
    }
}
