import QtQuick
import QtQuick.Controls.Material

ApplicationWindow {
    title: "Agent Dashboard"
    width: 800
    height: 450
    visible: true
    Material.theme: Material.Dark
    x: screen.desktopAvailableWidth - width - 12
    y: screen.desktopAvailableHeight - height - 48


    Loader {
        id: mainLoader
        anchors.fill: parent
        source: "Home.qml"
    }
}
