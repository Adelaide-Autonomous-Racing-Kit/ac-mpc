import QtQuick
import QtQuick.Layouts
import QtQuick.Controls.Material

Rectangle {
        id: sessionSummaryRoot
        color: "transparent"
        anchors{
            top: parent.top
            topMargin: 15
            left: parent.left
            leftMargin: 15
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
            id: currentLap
            width: parent.width
            anchors{
                top: lapsCompleted.bottom
                topMargin: 20
            }
            TextBody{
                Layout.alignment: Qt.AlignLeft
                text: "Current Lap"
            }
        }

        Rectangle{
            id: currentLapTimes
            color: "transparent"
            anchors{
                top: currentLap.bottom
                topMargin: 5
                left: parent.left
                leftMargin: 20
            }
            width: parent.width
            height: 80
            Item{
                id: currentLaptime
                width: parent.width/4
                anchors.left: parent.left
                TextBody{
                    id: lapTimeTitle
                    text: "Time"
                }
                TextBody {
                    id: currentLaptimeString
                    anchors.top: lapTimeTitle.bottom
                    text: sessionInfo.current_laptime
                    color: sessionInfo.current_lap_colour
                }
                TextBody {
                   anchors.top: currentLaptimeString.bottom
                   text: sessionInfo.current_lap_delta
                   color: sessionInfo.current_lap_delta_colour
                }
            }
            Item{
                id: firstSectorTime
                width: parent.width/4
                anchors.left: currentLaptime.right
                TextBody{
                    id: firstSectorTitle
                    text: "Sector 1"
                }
                TextBody {
                    id: firstSectorTimeString
                    anchors.top: firstSectorTitle.bottom
                    text: sessionInfo.current_first_sector_time
                    color: sessionInfo.first_sector_colour
                }
                TextBody {
                   anchors.top: firstSectorTimeString.bottom
                   text: sessionInfo.current_first_sector_delta
                   color: sessionInfo.current_first_sector_delta_colour
                }
            }
            Item{
                id: secondSectorTime
                width: parent.width/4
                anchors.left: firstSectorTime.right
                TextBody{
                    id: secondSectorTitle
                    text: "Sector 2"
                }
                TextBody {
                    id: secondSectorTimeString
                    anchors.top: secondSectorTitle.bottom
                    text: sessionInfo.current_second_sector_time
                    color: sessionInfo.second_sector_colour
                }
                TextBody {
                   anchors.top: secondSectorTimeString.bottom
                   text: sessionInfo.current_second_sector_delta
                   color: sessionInfo.current_second_sector_delta_colour
                }
            }
            Item{
                id: thirdSectorTime
                width: parent.width/4
                anchors.left: secondSectorTime.right
                TextBody{
                    id: thirdSectorTitle
                    text: "Sector 3"
                }
                TextBody {
                    id: thirdSectorTimeString
                    anchors.top: thirdSectorTitle.bottom
                    text: sessionInfo.current_third_sector_time
                    color: sessionInfo.third_sector_colour
                }
                TextBody {
                   anchors.top: thirdSectorTimeString.bottom
                   text: sessionInfo.current_third_sector_delta
                   color: sessionInfo.current_third_sector_delta_colour
                }
            }

        }
       
        RowLayout{
            id: lastLap
            width: parent.width
            anchors.top: currentLapTimes.bottom
            TextBody{
                Layout.alignment: Qt.AlignLeft
                text: "Last Lap"
            }
        }
        Rectangle{
            id: lastLaptimes
            color: "transparent"
            anchors{
                top: lastLap.bottom
                topMargin: 5
                left: parent.left
                leftMargin: 20
            }
            width: parent.width
            height: 80
            Item{
                id: lastLaptime
                width: parent.width/4
                anchors.left: parent.left
                TextBody{
                    id: lastLapTimeTitle
                    text: "Time"
                }
                TextBody {
                    id: lastLapTimeString
                    anchors.top: lastLapTimeTitle.bottom
                    text: sessionInfo.last_laptime
                    color: sessionInfo.last_lap_colour
                }
                TextBody {
                   anchors.top: lastLapTimeString.bottom
                   text: sessionInfo.last_lap_delta
                   color: sessionInfo.last_lap_delta_colour
                }
            }
            Item{
                id: lastFirstSectorTime
                width: parent.width/4
                anchors.left: lastLaptime.right
                TextBody{
                    id: lastFirstSectorTitle
                    text: "Sector 1"
                }
                TextBody {
                    id: lastFirstSectorTimeString
                    anchors.top: lastFirstSectorTitle.bottom
                    text: sessionInfo.last_first_sector_time
                    color: sessionInfo.last_first_sector_colour
                }
                TextBody {
                   anchors.top: lastFirstSectorTimeString.bottom
                   text: sessionInfo.last_first_sector_delta
                   color: sessionInfo.last_first_sector_delta_colour
                }
            }
            Item{
                id: lastSecondSectorTime
                width: parent.width/4
                anchors.left: lastFirstSectorTime.right
                TextBody{
                    id: lastSecondSectorTitle
                    text: "Sector 2"
                }
                TextBody {
                    id: lastSecondSectorTimeString
                    anchors.top: lastSecondSectorTitle.bottom
                    text: sessionInfo.last_second_sector_time
                    color: sessionInfo.last_second_sector_colour
                }
                TextBody {
                   anchors.top: lastSecondSectorTimeString.bottom
                   text: sessionInfo.last_second_sector_delta
                   color: sessionInfo.last_second_sector_delta_colour
                }
            }
            Item{
                id: lastThirdSectorTime
                width: parent.width/4
                anchors.left: lastSecondSectorTime.right
                TextBody{
                    id: lastThirdSectorTitle
                    text: "Sector 3"
                }
                TextBody {
                    id: lastThirdSectorTimeString
                    anchors.top: lastThirdSectorTitle.bottom
                   text: sessionInfo.last_third_sector_time
                    color: sessionInfo.last_third_sector_colour
                }
                TextBody {
                   anchors.top: lastThirdSectorTimeString.bottom
                   text: sessionInfo.last_third_sector_delta
                   color: sessionInfo.last_third_sector_delta_colour
                }
            }

        }

        RowLayout{
            id: bestLaptime
            width: parent.width
            anchors.top: lastLaptimes.bottom
            TextBody{
                Layout.alignment: Qt.AlignLeft
                text: qsTr("Best Lap ") + sessionInfo.best_lap
            }
        }
        Rectangle{
            id: bestLapTimes
            color: "transparent"
            anchors{
                top: bestLaptime.bottom
                topMargin: 5
                left: parent.left
                leftMargin: 20
            }
            width: parent.width
            height: 80
            Item{
                id: bestLapTime
                width: parent.width/4
                anchors.left: parent.left
                TextBody{
                    id: bestLapTimeTitle
                    text: "Time"
                }
                TextBody {
                    anchors.top: bestLapTimeTitle.bottom
                    text: sessionInfo.best_laptime
                }
            }
            Item{
                id: bestFirstSectorTime
                width: parent.width/4
                anchors.left: bestLapTime.right
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
                width: parent.width/4
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
                width: parent.width/4
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