#!/bin/sh
COMPAT_TOOL=$1

if [ $COMPAT_TOOL = "proton" ]; then
    SETUPS_PATH="$HOME/.local/share/Steam/steamapps/compatdata/244210/pfx/drive_c/users/steamuser/Documents/Assetto Corsa/setups"
elif [ $COMPAT_TOOL = "crossover" ]; then
    SETUPS_PATH="$HOME/Documents/Assetto Corsa/setups"
else
    echo "Provided compatibility tool argument: \"$COMPAT_TOOL\" is invalid.";
    echo "Valid options are \"proton\" or \"corssover\"";
    exit 0;
fi

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
maybe_download_asset "$MODELS_PATH/vallelunga-fpn-resnet-18-v1.2.pt" "$HOST_URL/models/segmentation/vallelunga/vallelunga-fpn-resnet-18-v1.2.pt"
maybe_download_asset "$MODELS_PATH/yas_marina-fpn-resnet-18-v1.pt" "$HOST_URL/models/segmentation/yas-marina/yas_marina-fpn-resnet-18-v1.pt"
maybe_download_asset "$MODELS_PATH/nordschleife-fpn-resnet-18-v1.pt" "$HOST_URL/models/segmentation/nordschleife/nordschleife-fpn-resnet-18-v1.pt"
# Maps
maybe_download_asset "$MAPS_PATH/monza_verysmooth.npy" "$HOST_URL/maps/monza/monza_verysmooth.npy"
maybe_download_asset "$MAPS_PATH/spa_verysmooth.npy" "$HOST_URL/maps/spa/spa_verysmooth.npy"
maybe_download_asset "$MAPS_PATH/silverstone_verysmooth.npy" "$HOST_URL/maps/silverstone/silverstone_verysmooth.npy"
maybe_download_asset "$MAPS_PATH/vallelunga_verysmooth.npy" "$HOST_URL/maps/vallelunga/vallelunga_verysmooth.npy"
# Setups
# Audi R8 LMS 2016
AUDI_R8_SETUPS_PATH="$SETUPS_PATH/ks_audi_r8_lms_2016"
# Make folders
mkdir -p "$AUDI_R8_SETUPS_PATH/monza"
mkdir -p "$AUDI_R8_SETUPS_PATH/spa"
mkdir -p "$AUDI_R8_SETUPS_PATH/ks_nordschleife"
mkdir -p "$AUDI_R8_SETUPS_PATH/ks_silverstone"
mkdir -p "$AUDI_R8_SETUPS_PATH/ks_vallelunga"
mkdir -p "$AUDI_R8_SETUPS_PATH/abudhabi_euroracers_v2"
# Download setups
maybe_download_asset "$AUDI_R8_SETUPS_PATH/monza/0.ini" "$HOST_URL/setups/audi-r8-lms-2016/monza/0.ini"
maybe_download_asset "$AUDI_R8_SETUPS_PATH/spa/0.ini" "$HOST_URL/setups/audi-r8-lms-2016/spa/0.ini"
maybe_download_asset "$AUDI_R8_SETUPS_PATH/ks_silverstone/0.ini" "$HOST_URL/setups/audi-r8-lms-2016/silverstone/0.ini"
maybe_download_asset "$AUDI_R8_SETUPS_PATH/ks_vallelunga/0.ini" "$HOST_URL/setups/audi-r8-lms-2016/vallelunga/0.ini"
# maybe_download_asset "$AUDI_R8_SETUPS_PATH/abudhabi_euroracers_v2/0.ini" "$HOST_URL/setups/audi-r8-lms-2016/yas-marina/0.ini"

