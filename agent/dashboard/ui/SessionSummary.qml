import QtQuick
import QtQuick.Layouts
import QtQuick.Controls.Material

Rectangle {
        id: sessionSummaryRoot
        color: "transparent"
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

        RowLayout{
            id: lapsCompleted
            width: parent.width
            anchors.top: parent.top
            TextHeading{
                Layout.alignment: Qt.AlignLeft
                text: qsTr("Lap ") + sessionInfo.n_laps_completed
            }
        }

        RowLayout{
            id: currentLaptime
            width: parent.width
            anchors.top: lapsCompleted.bottom
            TextBody{
                Layout.alignment: Qt.AlignLeft
                text: qsTr("Current ") + sessionInfo.current_laptime
            }
        }

        Rectangle{
            id: currentSectorTimes
            color: "transparent"
            anchors.top: currentLaptime.bottom
            width: parent.width
            height: 70
            Item{
                id: firstSectorTime
                width: parent.width/3
                anchors.left: parent.left
                TextBody{
                    id: firstSectorTitle
                    text: "Sector 1"
                }
                TextBody {
                    anchors.top: firstSectorTitle.bottom
                    text: sessionInfo.current_first_sector_time
                    color: sessionInfo.first_sector_colour
                }
            }
            Item{
                id: secondSectorTime
                width: parent.width/3
                anchors.left: firstSectorTime.right
                TextBody{
                    id: secondSectorTitle
                    text: "Sector 2"
                }
                TextBody {
                    anchors.top: secondSectorTitle.bottom
                    text: sessionInfo.current_second_sector_time
                    color: sessionInfo.second_sector_colour
                }
            }
            Item{
                id: thirdSectorTime
                width: parent.width/3
                anchors.left: secondSectorTime.right
                TextBody{
                    id: thirdSectorTitle
                    text: "Sector 3"
                }
                TextBody {
                    anchors.top: thirdSectorTitle.bottom
                    text: sessionInfo.current_third_sector_time
                    color: sessionInfo.third_sector_colour
                }
            }

        }
       
        RowLayout{
            id: lastLaptime
            width: parent.width
            anchors.top: currentSectorTimes.bottom
            TextBody{
                Layout.alignment: Qt.AlignLeft
                text: qsTr("Last ") + sessionInfo.last_laptime
            }
        }
        Rectangle{
            id: lastSectorTimes
            color: "transparent"
            anchors.top: lastLaptime.bottom
            width: parent.width
            height: 70
            Item{
                id: lastFirstSectorTime
                width: parent.width/3
                anchors.left: parent.left
                TextBody{
                    id: lastFirstSectorTitle
                    text: "Sector 1"
                }
                TextBody {
                    anchors.top: lastFirstSectorTitle.bottom
                    text: sessionInfo.last_first_sector_time
                    color: sessionInfo.last_first_sector_colour
                }
            }
            Item{
                id: lastSecondSectorTime
                width: parent.width/3
                anchors.left: lastFirstSectorTime.right
                TextBody{
                    id: lastSecondSectorTitle
                    text: "Sector 2"
                }
                TextBody {
                    anchors.top: lastSecondSectorTitle.bottom
                    text: sessionInfo.last_second_sector_time
                    color: sessionInfo.last_second_sector_colour
                }
            }
            Item{
                id: lastThirdSectorTime
                width: parent.width/3
                anchors.left: lastSecondSectorTime.right
                TextBody{
                    id: lastThirdSectorTitle
                    text: "Sector 3"
                }
                TextBody {
                    anchors.top: lastThirdSectorTitle.bottom
                    text: sessionInfo.last_third_sector_time
                    color: sessionInfo.last_third_sector_colour
                }
            }

        }

        RowLayout{
            id: bestLaptime
            width: parent.width
            anchors.top: lastSectorTimes.bottom
            TextBody{
                Layout.alignment: Qt.AlignLeft
                text: qsTr("Best ") + sessionInfo.best_laptime
            }
        }
        Rectangle{
            id: bestSectorTimes
            color: "transparent"
            anchors.top: bestLaptime.bottom
            width: parent.width
            height: 70
            Item{
                id: bestFirstSectorTime
                width: parent.width/3
                anchors.left: parent.left
                TextBody{
                    id: bestFirstSectorTitle
                    text: "Sector 1"
                }
                TextBody {
                    anchors.top: bestFirstSectorTitle.bottom
                    text: sessionInfo.best_first_sector_time
                }
            }
            Item{
                id: bestSecondSectorTime
                width: parent.width/3
                anchors.left: bestFirstSectorTime.right
                TextBody{
                    id: bestSecondSectorTitle
                    text: "Sector 2"
                }
                TextBody {
                    anchors.top: bestSecondSectorTitle.bottom
                    text: sessionInfo.best_second_sector_time
                }
            }
            Item{
                id: bestThirdSectorTime
                width: parent.width/3
                anchors.left: bestSecondSectorTime.right
                TextBody{
                    id: bestThirdSectorTitle
                    text: "Sector 3"
                }
                TextBody {
                    anchors.top: bestThirdSectorTitle.bottom
                    text: sessionInfo.best_third_sector_time
                }
            }

        }

       
}