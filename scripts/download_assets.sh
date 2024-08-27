#!/bin/sh
maybe_download_asset () {
    FILE=$1
    URL=$2
    if [ ! -f "$FILE" ]; then
        echo "Downloading $FILE from $URL"
        curl -L -o "$FILE" "$URL"
    fi
}

# Download assets
HOST_URL=https://adelaideautonomous.racing/store/aarc
MODELS_PATH=data/models/segmentation
MAPS_PATH=data/maps
# Vehicle Data
if [ ! -d "data/vehicles/audi_r8_lms_2016/" ]; then
    maybe_download_asset "data/vehicles/audi_r8_lms_2016.zip" "$HOST_URL/cars/audi_r8_lms_2016.zip"
    unzip data/vehicles/audi_r8_lms_2016.zip -d data/vehicles/
    rm data/vehicles/audi_r8_lms_2016.zip
fi
# Segmentation Models
maybe_download_asset "$MODELS_PATH/monza-fpn-resnet-18-v1.1.pt" "$HOST_URL/models/segmentation/monza/monza-fpn-resnet-18-v1.1.pt"
maybe_download_asset "$MODELS_PATH/spa-fpn-resnet-18-v1.pt" "$HOST_URL/models/segmentation/spa/spa-fpn-resnet-18-v1.pt"
maybe_download_asset "$MODELS_PATH/silverstone-fpn-resnet-18-v1.1.pt" "$HOST_URL/models/segmentation/silverstone/silverstone-fpn-resnet-18-v1.1.pt"
maybe_download_asset "$MODELS_PATH/vallelunga-fpn-resnet-18-v1.1.pt" "$HOST_URL/models/segmentation/vallelunga/vallelunga-fpn-resnet-18-v1.1.pt"
maybe_download_asset "$MODELS_PATH/yas_marina-fpn-resnet-18-v1.pt" "$HOST_URL/models/segmentation/yas-marina/yas_marina-fpn-resnet-18-v1.pt"
# Maps
maybe_download_asset "$MAPS_PATH/monza_verysmooth.npy" "$HOST_URL/maps/monza/monza_verysmooth.npy"
maybe_download_asset "$MAPS_PATH/spa_verysmooth.npy" "$HOST_URL/maps/spa/spa_verysmooth.npy"
maybe_download_asset "$MAPS_PATH/silverstone_verysmooth.npy" "$HOST_URL/maps/silverstone/silverstone_verysmooth.npy"
maybe_download_asset "$MAPS_PATH/vallelunga_verysmooth.npy" "$HOST_URL/maps/vallelunga/vallelunga_verysmooth.npy"
# Setups
# Make folders