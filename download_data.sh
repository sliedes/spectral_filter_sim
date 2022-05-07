#!/bin/bash

set -xe

DATA=(
    balloons_ms beads_ms cd_ms cloth_ms clay_ms egyptian_statue_ms feathers_ms flowers_ms
    glass_tiles_ms chart_and_stuffed_toy_ms pompoms_ms sponges_ms thread_spools_ms
    stuffed_toys_ms superballs_ms
    face_ms photo_and_face_ms hairs_ms
    oil_painting_ms paints_ms #watercolors_ms
    beers_ms jelly_beans_ms lemon_slices_ms lemons_ms peppers_ms strawberries_ms sushi_ms
    tomatoes_ms yellowpeppers_ms)

mkdir -p data

for img in "${DATA[@]}"; do
    if [ ! -d data/$img ]; then
        rm -f data.zip
        curl https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/$img.zip -o data.zip
        (cd data && unzip -o ../data.zip </dev/null)
        rm -f data.zip
    fi
done

find data \( -name \*.bmp -or -name \*.db \) -delete
