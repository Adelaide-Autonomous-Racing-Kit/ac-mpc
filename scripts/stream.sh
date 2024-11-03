#!/bin/bash
while true;
do
python main.py --config configs/monza.yaml
python main.py --config configs/spa.yaml
python main.py --config configs/silverstone.yaml
python main.py --config configs/vallelunga.yaml
python main.py --config configs/bathurst.yaml
done