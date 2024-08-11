import QtQuick
import QtQuick.Controls.Material

ApplicationWindow {
    title: "Agent Dashboard"
    width: 1920
    height: 1080
    visible: true
    Material.theme: Material.Dark
    // Start in top right of desktop
    x: screen.desktopAvailableWidth - width
    y: 0


    Loader {
        id: mainLoader
        anchors.fill: parent
        source: "HomeStream.qml"
    }
}
