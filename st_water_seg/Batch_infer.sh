#!/bin/bash

# Root folder containing the directories
# Set root folder to current direcotry
ROOT_FOLDER=$(pwd)
#ROOT_FOLDER="/media/mule/Projects/SocialPixelLab/RGV/Event1_DamBreak_2021-2/"

# JSON file path
JSON_FILE="/dataset_dirs.json"

# Loop through each sub-directory in the root folder
for dir in "$ROOT_FOLDER"/*; do
    if [ -d "$dir" ]; then
        # Extract the folder name
        FOLDER_NAME="$dir"

        # Update the JSON file
        jq --arg folder_name "$FOLDER_NAME" '.thp_timeseries = $folder_name' "$JSON_FILE" > "$JSON_FILE.tmp" && mv "$JSON_FILE.tmp" "$JSON_FILE"

        # Execute the command
        CUDA_VISIBLE_DEVICES=1 python ./st_water_seg/infer.py /home/zhijiezhang/spatial_temporal_water_seg/Trained_models_NeurIPS/PS_high+low_aug_300_b8_try2/LittleRock/checkpoints/model-epoch=09-val_MulticlassJaccardIndex=0.8760.ckpt thp_timeseries all Pre pre
    fi
done