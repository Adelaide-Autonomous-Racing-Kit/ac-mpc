#!/bin/bash
while true;
do
bash scripts/run.sh /configs/monza.yaml
bash scripts/run.sh /configs/spa.yaml
bash scripts/run.sh /configs/silverstone.yaml
bash scripts/run.sh /configs/vallelunga.yaml
done