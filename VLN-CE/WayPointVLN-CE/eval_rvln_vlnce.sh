#!/bin/bash

CHUNKS=1

################ NaVid ################

MODEL_PATH="/home/isvl/guan_code/WayPoint-VLN/output/rvln_merged_final" 
MODEL_NAME="rvln" # uni-navid or navid
EXP_SAVE="video-data" # use "data" to accelerate evaluation

#R2R
CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
SAVE_PATH="tmp/rvln" 


#RxR
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/rxr_baselines/navid_rxr.yaml"
# SAVE_PATH="tmp/results_navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split-rxr" 


################ Uni-NaVid ################

# MODEL_PATH="model_zoo/llama-vid-7b-full-224-video-fps-1-grid-2-panda-encoder-2025-10-12-all-data"
# MODEL_NAME="uni-navid" # uni-navid or navid
# EXP_SAVE="video-data" # use "data" to accelerate evaluation

#R2R
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_r2r.yaml"
# SAVE_PATH="tmp/results_uninavid-7b-full-224-video-fps-1-grid-2-r2r" 


#RxR
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/rxr_baselines/uninavid_rxr.yaml"
# SAVE_PATH="tmp/results_uninavid-7b-full-224-video-fps-1-grid-2-rxr" 




for IDX in $(seq 0 $((CHUNKS-1))); do
    echo $(( IDX % 8 ))
    CUDA_VISIBLE_DEVICES=$(( IDX % 8 )) python run.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --model-path $MODEL_PATH \
    --result-path $SAVE_PATH \
    --model-name $MODEL_NAME \
    --exp-save $EXP_SAVE &
    
done

wait

