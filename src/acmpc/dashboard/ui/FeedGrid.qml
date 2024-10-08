import QtQuick
import QtQuick.Controls.Material
import QtQuick.Layouts

GridLayout {
    id: gridRoot
    anchors{
        bottom: parent.bottom
        left: parent.left
        right: parent.right
        leftMargin: 15
        rightMargin: 15
        topMargin: 5
        bottomMargin: 15
    }
    columns: 3
    rows: 3

    Rectangle {
        color: "transparent"
        border.color: "slategray"
        Layout.fillHeight: true
        Layout.fillWidth: true
        FeedWindow{
            id: cameraImage
            anchors.bottom: toggleCameraFeed.top
            source_root: "image://cameraFeed/img"
            Connections{
                target: cameraFeed

                function onImageChanged(image) {
                    cameraImage.reloadImage()
                }
            }
        }
        MyButton {
            id: toggleCameraFeed
            anchors{
                bottom: parent.bottom
                horizontalCenter: parent.horizontalCenter
            }

            property bool is_started: false
            text: is_started ? "Stop Camera" : "Start Camera" 
            onClicked: {
                if (is_started){
                    cameraFeed.shutdown()
                } else {
                    cameraFeed.start()
                }
                is_started = !is_started
            }
        }
    }
    Rectangle {
        color: "transparent"
        border.color: "slategray"
        Layout.fillHeight: true
        Layout.fillWidth: true
        FeedWindow{
            id: segmentationImage
            anchors.bottom: toggleSegmentationFeed.top

            source_root: "image://segmentationFeed/img"
            Connections{
                target: segmentationFeed

                function onImageChanged(image) {
                    segmentationImage.reloadImage()
                }
            }
        }
        MyButton {
            id: toggleSegmentationFeed
            anchors{
                bottom: parent.bottom
                horizontalCenter: parent.horizontalCenter
            }

            property bool is_started: false
            text: is_started ? "Stop Masks" : "Start Masks"
            onClicked: {
                if (is_started){
                    segmentationFeed.shutdown()
                } else {
                    segmentationFeed.start()
                }
                is_started = !is_started
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
            anchors.bottom: toggleControlFeed.top

            source_root: "image://controlFeed/img"
            Connections{
                target: controlFeed

                function onImageChanged(image) {
                    controlImage.reloadImage()
                }
            }
        }
        MyButton {
            id: toggleControlFeed
            anchors{
                bottom: parent.bottom
                horizontalCenter: parent.horizontalCenter
            }

            property bool is_started: false
            text: is_started ?  "Stop Control" : "Start Control"
            onClicked: {
                if (is_started){
                    controlFeed.shutdown()
                } else {
                    controlFeed.start()
                }
                is_started = !is_started
             }
        }
    }
    Rectangle {
        color: "transparent"
        border.color: "slategray"
        Layout.fillHeight: true
        Layout.fillWidth: true
        FeedWindow{
            id: predictionsImage
            anchors.bottom: togglePredictionsFeed.top

            source_root: "image://predictionsFeed/img"
            Connections{
                target: predictionsFeed

                function onImageChanged(image) {
                    predictionsImage.reloadImage()
                }
            }
        }
        MyButton {
            id: togglePredictionsFeed
            anchors{
                bottom: parent.bottom
                horizontalCenter: parent.horizontalCenter
            }

            property bool is_started: false
            text: is_started ? "Stop Predictions":"Start Predictions"
            onClicked: {
                if (is_started){
                    predictionsFeed.shutdown()
                } else {
                    predictionsFeed.start()
                }
                is_started = !is_started
             }
        }
    }
    Rectangle {
        color: "transparent"
        border.color: "slategray"
        Layout.fillHeight: true
        Layout.fillWidth: true
        FeedWindow{
            id: localLocalisationImage
            anchors.bottom: toggleLocalLocalisationFeed.top

            source_root: "image://localLocalisationFeed/img"
            Connections{
                target: localLocalisationFeed

                function onImageChanged(image) {
                    localLocalisationImage.reloadImage()
                }
            }
        }
        MyButton {
            id: toggleLocalLocalisationFeed
            anchors{
                bottom: parent.bottom
                horizontalCenter: parent.horizontalCenter
            }

            property bool is_started: false
            text: is_started ? "Stop Localisation":"Start Localisation"
            onClicked: {
                if (is_started){
                    localLocalisationFeed.shutdown()
                } else {
                    localLocalisationFeed.start()
                }
                is_started = !is_started
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
            anchors.bottom: toggleMapLocalisationFeed.top
            
            source_root: "image://mapLocalisationFeed/img"
            Connections{
                target: mapLocalisationFeed

                function onImageChanged(image) {
                    mapLocalisationImage.reloadImage()
                }
            }
        }
        MyButton {
            id: toggleMapLocalisationFeed
            anchors{
                bottom: parent.bottom
                horizontalCenter: parent.horizontalCenter
            }

            property bool is_started: false
            text: is_started ?  "Stop Map" : "Start Map"
            onClicked: {
                if (is_started){
                    mapLocalisationFeed.shutdown()
                } else {
                    mapLocalisationFeed.start()
                }
                is_started = !is_started
             }
        }
    }
}