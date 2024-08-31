import QtQuick
import QtQuick.Controls.Material

ApplicationWindow {
    title: "Agent Dashboard"
    width: 800
    height: 450
    visible: true
    Material.theme: Material.Dark


    Loader {
        id: mainLoader
        anchors.fill: parent
        source: "Home.qml"
    }
}
