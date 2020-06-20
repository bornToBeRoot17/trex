#!/bin/bash

# MODE = 0: train_size resized to test_size; test_size resized to train_size
# MODE = 1: train/test with no resize, no extraction. Just copy
# MODE = 2: individually train/test (complete steps), no resize
# MODE = 3: individually train/test (complete steps), resize test to train
# MODE = 4: just extraction, no train/test (useful for update extracted paths)

MODE=3
#TRAIN_SIZE=(32 64 100 128 200 256 300)
#TEST_SIZE=(32 64 100 128 200 256 300)
TRAIN_SIZE=(128)
TEST_SIZE=(256)
#CLASSES=("ft" "lu" "map" "allreduce")
#CLASSES=("multc" "cylinder3d") # "lu" "map" "allreduce")
CLASSES=("bifu" "cyl2d" "multc")
EXTRACTORS=("DCTraCS_RLBP" "DCTraCS_ULBP" "Eerman" "Soysal" "Zhang" "LBP") # "Fahad"
RESULT_DIR="./results_nof/"
#PARSER="image_read_dilation"
PARSER="image_parser"
SCK_PATH="./training_scikit_out/"
DATASET="/nobackup/ppginf/rgcastro/research/trex_dataset/mpi_dataset/"
RESULT_SUFIX="lenet"

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

NUM_IMGS=400
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

if [ $MODE -eq 3 ]; then
    STR_CLASSES=""
    for i in "${CLASSES[@]}"; do
        STR_CLASSES="$STR_CLASSES${i},"
    done;
    STR_CLASSES="${STR_CLASSES::-1}"

    for i in "${TRAIN_SIZE[@]}"; do
        for j in "${TEST_SIZE[@]}"; do
            RESULT_FILE="${RESULT_DIR}result_tr${i}_ts${j}_${STR_CLASSES}_n${NUM_IMGS}.txt"
            count_file=1
            while [ -f "$RESULT_FILE" ]; do
                RESULT_FILE="${RESULT_DIR}result_tr${i}_ts${j}_${STR_CLASSES}_n${NUM_IMGS}_"${count_file}".txt"
                count_file=$((count_file+1))
            done;
            python3 generate_definitions.py "["${i}"]" "[0]" "["${j}"]" "["${i}"]" ${NUM_IMGS} ${PARSER} ${SCK_PATH} ${STR_CLASSES} ${DATASET} &&
            python3 lenet_2.py "tr${i}_r0_ts${j}_r${i}" &&

            #rm -r ${SCK_PATH}
            echo $RESULT_FILE
        done;
    done;
fi;
