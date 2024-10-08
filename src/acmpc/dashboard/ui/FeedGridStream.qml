import QtQuick
import QtQuick.Controls.Material
import QtQuick.Layouts

GridLayout {
    id: gridRoot
    anchors{
        bottom: parent.bottom
        left: parent.left
        right: parent.right
        leftMargin: 5
        rightMargin: 5
        topMargin: 5
        bottomMargin: 5
    }
    columns: 3
    rows: 3

    
    Rectangle {
        color: "transparent"
        border.color: "slategray"
        Layout.fillHeight: false
        Layout.preferredHeight: 392
        Layout.fillWidth: true
        SessionSummary{}
    }
    Rectangle {
        color: "transparent"
        Layout.fillHeight: false
        Layout.preferredHeight: 392
        Layout.fillWidth: true
    }
    Rectangle {
        color: "transparent"
        Layout.fillHeight: false
        Layout.preferredHeight: 392
        Layout.fillWidth: true
    }

    Rectangle {
        color: "transparent"
        border.color: "slategray"
        Layout.fillHeight: true
        Layout.fillWidth: true
        FeedWindow{
            id: predictionsImage
            anchors{
                bottom: parent.bottom
                bottomMargin: 5
            }

            source_root: "image://predictionsFeed/img"
            Connections{
                target: predictionsFeed

                function onImageChanged(image) {
                    predictionsImage.reloadImage()
                }
            }
            Component.onCompleted: {
                predictionsFeed.start()
            }
        }
    }
    Rectangle {
        color: "transparent"
        Layout.fillHeight: true
        Layout.fillWidth: true
    }
    Rectangle {
        color: "transparent"
        Layout.fillHeight: true
        Layout.fillWidth: true
    }

    Rectangle {
        color: "transparent"
        border.color: "slategray"
        Layout.fillHeight: true
        Layout.fillWidth: true
        FeedWindow{
            id: segmentationImage
            anchors{
                bottom: parent.bottom
                bottomMargin: 5
            }

            source_root: "image://segmentationFeed/img"
            Connections{
                target: segmentationFeed

                function onImageChanged(image) {
                    segmentationImage.reloadImage()
                }
            }
            Component.onCompleted: {
                segmentationFeed.start()
            }
        }
    }
    Rectangle {
        color: "transparent"
        border.color: "slategray"
        Layout.fillHeight: true
        Layout.fillWidth: true
        FeedWindow{
            id: controlImage
            anchors{
                bottom: parent.bottom
                bottomMargin: 5
            }

            source_root: "image://controlFeed/img"
            Connections{
                target: controlFeed

                function onImageChanged(image) {
                    controlImage.reloadImage()
                }
            }
            Component.onCompleted: {
                controlFeed.start()
            }
        }
    }
    Rectangle {
        color: "transparent"
        border.color: "slategray"
        Layout.fillHeight: true
        Layout.fillWidth: true
        FeedWindow{
            id: mapLocalisationImage
            anchors{
                bottom: parent.bottom
                bottomMargin: 5
            }
            
            source_root: "image://mapLocalisationFeed/img"
            Connections{
                target: mapLocalisationFeed

                function onImageChanged(image) {
                    mapLocalisationImage.reloadImage()
                }
            }
            Component.onCompleted: {
                mapLocalisationFeed.start()
            }
        }
    }

}