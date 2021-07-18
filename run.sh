#!/bin/bash

# MODE = 0: train_size resized to test_size; test_size resized to train_size
# MODE = 1: train/test with no resize, no extraction. Just copy
# MODE = 2: individually train/test (complete steps), no resize
# MODE = 3: individually train/test (complete steps), resize test to train
# MODE = 4: just extraction, no train/test (useful for update extracted paths)

#MODE=3
##TRAIN_SIZE=(32 64 100 128 200 256 300)
##TEST_SIZE=(32 64 100 128 200 256 300)
#TRAIN_SIZE=(128)
#TEST_SIZE=(256)
##CLASSES=("ft" "lu" "map" "allreduce")
##CLASSES=("multc" "cylinder3d") # "lu" "map" "allreduce")
##CLASSES=("bifu" "cyl2d" "multc" "cylinder3d") #"ft" "lu" "map" "allreduce")
#CLASSES=("bifu" "cyl2d" "multc") #"ft" "lu" "map" "allreduce")
#EXTRACTORS=("DCTraCS_RLBP" "DCTraCS_ULBP" "Eerman" "Soysal" "Zhang" "LBP" "GLCM") # "Fahad"
#PARSER="image_read_dilation"
#SCK_PATH="./training_scikit_out/"
#DATASET="../trex_dataset/aug_real_dataset/"
#RESULT_SUFIX="glcm_resize_dilation"
#RESULT_DIR="./results_${RESULT_SUFIX}/"

# Synthetic
MODE=3
TRAIN_SIZE=(128)
TEST_SIZE=(128)
CLASSES=("ft" "lu" "map" "allreduce")
EXTRACTORS=("DCTraCS_RLBP" "DCTraCS_ULBP" "Eerman" "Soysal" "Zhang" "LBP" "GLCM")
PARSER="image_parser"
SCK_PATH="./training_scikit_out/"
DATASET="../trex_dataset/dataset_synthetic/"
RESULT_SUFIX="1121"
RESULT_DIR="./results_master/"

if [ ! -d "${RESULT_DIR}" ]; then
    mkdir ${RESULT_DIR}
fi;

if [ -d "${SCK_PATH}" ]; then
    rm -r ${SCK_PATH}
fi;
mkdir ${SCK_PATH}
for k in "${EXTRACTORS[@]}"; do
    mkdir ${SCK_PATH}${k}
done;

NUM_IMGS=300
for i in "${CLASSES[@]}"; do
    for j in "${TRAIN_SIZE[@]}"; do
        N="$(ls ${DATASET}${i}_${j} | wc -l)"
        if [ $N -lt $NUM_IMGS ]; then
            NUM_IMGS=$N
        fi;
    done;

    for j in "${TEST_SIZE[@]}"; do
        N="$(ls ${DATASET}${i}_${j} | wc -l)"
        if [ $N -lt $NUM_IMGS ]; then
            NUM_IMGS=$N
        fi;
    done;
done;
NUM_IMGS=$(( NUM_IMGS-1 ))

STR_CLASSES=""
for i in "${CLASSES[@]}"; do
    STR_CLASSES="$STR_CLASSES${i},"
done;
STR_CLASSES="${STR_CLASSES::-1}"

for k in `seq 1 10`; do
    for i in "${TRAIN_SIZE[@]}"; do
        for j in "${TEST_SIZE[@]}"; do
            rm -r ${SCK_PATH}
            mkdir ${SCK_PATH}
            for l in "${EXTRACTORS[@]}"; do
                mkdir ${SCK_PATH}${l}
            done;

            python3 generate_definitions.py "["${i}"]" "[0]" "["${j}"]" "[0]" ${NUM_IMGS} ${PARSER} ${SCK_PATH} ${STR_CLASSES} ${DATASET} &&
            python3 DCTraCSprocessing.py "tr${i}_r0_ts${j}_r0"

            RESULT_FILE="${RESULT_DIR}${RESULT_SUFIX}_tr${i}_ts${j}_${STR_CLASSES}_n${NUM_IMGS}.txt"
            count_file=1
            while [ -f "$RESULT_FILE" ]; do
                RESULT_FILE="${RESULT_DIR}${RESULT_SUFIX}_tr${i}_ts${j}_${STR_CLASSES}_n${NUM_IMGS}_"${count_file}".txt"
                count_file=$((count_file+1))
            done;
            python3 DCTraCSresults.py "tr${i}_r0_ts${j}_r0" > $RESULT_FILE

            #rm -r ${SCK_PATH}
            echo $RESULT_FILE
        done;
    done;
done;

