#!/bin/bash

# Real
MODE=3
TRAIN_SIZE=(128)
TEST_SIZE=(256)
#EXTRACTORS=("DCTraCS_RLBP" "DCTraCS_ULBP" "Fahad" "Soysal" "Eerman")
EXTRACTORS=("Fahad" "Soysal" "Eerman")
PARSER="image_parser"
SCK_PATH="./training_scikit_out/"
DATASET="../trex_dataset/128_256_dataset/"
RESULT_SUFIX="treco_three_real_classes_fahad_soysal_eerman"
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

CLASSES=()
for k in `ls ${DATASET} | cut -d_ -f1 | sort --unique`; do
    CLASSES+=( $k )
done;
CLASSES=("bifu" "multc" "cyl2d")

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

            python3 generate_definitions.py "["${i}"]" "[0]" "["${j}"]" "["${i}"]" ${NUM_IMGS} ${PARSER} ${SCK_PATH} ${STR_CLASSES} ${DATASET} &&
            python3 DCTraCSprocessing.py "tr${i}_r0_ts${j}_r${i}"

            RESULT_FILE="${RESULT_DIR}${RESULT_SUFIX}_tr${i}_ts${j}_${STR_CLASSES}_n${NUM_IMGS}.txt"
            count_file=1
            while [ -f "$RESULT_FILE" ]; do
                RESULT_FILE="${RESULT_DIR}${RESULT_SUFIX}_tr${i}_ts${j}_${STR_CLASSES}_n${NUM_IMGS}_"${count_file}".txt"
                count_file=$((count_file+1))
            done;
            python3 DCTraCSresults.py "tr${i}_r0_ts${j}_r${i}" > $RESULT_FILE

            #rm -r ${SCK_PATH}
            echo $RESULT_FILE
        done;
    done;
done;
