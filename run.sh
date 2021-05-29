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
CLASSES=("bifu" "cyl2d" "multc" "cylinder3d") #"ft" "lu" "map" "allreduce")
EXTRACTORS=("DCTraCS_RLBP" "DCTraCS_ULBP" "Eerman" "Soysal" "Zhang" "LBP") # "Fahad"
PARSER="normal_parser"
#PARSER="image_parser"
SCK_PATH="./training_scikit_out/"
DATASET="../trex_dataset/aug_real_dataset/"
RESULT_SUFIX="glcm"
RESULT_DIR="./results_${RESULT_SUFIX}/"

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

if [ $MODE -eq 0 ]; then
    if [ ! -d "./results_resize_train/" ]; then
        mkdir "./results_resize_train/"
    fi;

    if [ ! -d "./results_resize_test/" ]; then
        mkdir "./results_resize_test/"
    fi;

    for i in "${TRAIN_SIZE[@]}"; do
        for j in "${TEST_SIZE[@]}"; do
            echo "tr${i}_r0_ts${j}_r${i}"

            python3 generate_definitions.py "["${i}"]" "[0]" "["${j}"]" "["${i}"]" ${NUM_IMGS} ${PARSER} ${SCK_PATH} ${STR_CLASSES} ${DATASET} &&
            python3 DCTraCSprocessing.py "tr${i}_r0_ts${j}_r${i}" &&
            python3 DCTraCSresults.py "tr${i}_r0_ts${j}_r${i}" > results_resize_test/resize_tr${i}_ts${j}.txt &&

            rm -r ${SCK_PATH}
        done;
    done;

    for i in "${TRAIN_SIZE[@]}"; do
        for j in "${TEST_SIZE[@]}"; do
            echo "tr${i}_r${j}_ts${j}_r0"

            python3 generate_definitions.py "["${i}"]" "["${j}"]" "["${i}"]" "[0]" ${NUM_IMGS} ${PARSER} ${SCK_PATH} ${STR_CLASSES} ${DATASET} &&
            python3 DCTraCSprocessing.py "tr${i}_r${j}_ts${j}_r0" &&
            python3 DCTraCSresults.py "tr${i}_r${j}_ts${j}_r0" > results_resize_train/resize_tr${i}_ts${j}.txt &&

            rm -r ${SCK_PATH}
        done;
    done;

elif [ $MODE -eq 1 ]; then
    for i in "${TRAIN_SIZE[@]}"; do
        for j in "${TEST_SIZE[@]}"; do
            if [ $i -eq $j ]; then
                for k in "${EXTRACTORS[@]}"; do
                    num_train=1
                    num_test=5
                    for l in "${CLASSES[@]}"; do
                        head -n 210 "extracted_"${i}/${k}/${l}.sck > ${SCK_PATH}${k}/class${num_train}.sck
                        tail -n 90 "extracted_"${j}/${k}/${l}.sck > ${SCK_PATH}${k}/class${num_test}.sck
                        num_train=$((num_train+1))
                        num_test=$((num_test+1))
                    done;
                done;
            else
                for k in "${EXTRACTORS[@]}"; do
                    num_train=1
                    num_test=5
                    for l in "${CLASSES[@]}"; do
                        cp "extracted_"${i}/${k}/${l}.sck ${SCK_PATH}${k}/class${num_train}.sck
                        cp "extracted_"${j}/${k}/${l}.sck ${SCK_PATH}${k}/class${num_test}.sck
                        num_train=$((num_train+1))
                        num_test=$((num_test+1))
                    done;
                done;
            fi;

            RESULT_FILE="${RESULT_DIR}result_tr${i}_ts${j}.txt"
            python3 generate_definitions.py "["${i}"]" "[0]" "["${j}"]" "[0]" ${NUM_IMGS}  ${PARSER} ${SCK_PATH} ${STR_CLASSES} ${DATASET} &&
            python3 DCTraCSresults.py "tr${i}_r0_ts${j}_r0" > $RESULT_FILE

            echo $RESULT_FILE
        done;
    done;

elif [ $MODE -eq 2 ]; then
    for i in "${TRAIN_SIZE[@]}"; do
        for j in "${TEST_SIZE[@]}"; do
            RESULT_FILE="${RESULT_DIR}result_tr${i}_ts${j}.txt"
            python3 generate_definitions.py "["${i}"]" "[0]" "["${j}"]" "[0]" ${NUM_IMGS} ${PARSER} ${SCK_PATH} ${STR_CLASSES} ${DATASET} &&
            python3 DCTraCSprocessing.py "tr${i}_r0_ts${j}_r0" &&
            python3 DCTraCSresults.py "tr${i}_r0_ts${j}_r0" > $RESULT_FILE &&

            rm -r ${SCK_PATH}
            echo $RESULT_FILE
        done;
    done;

elif [ $MODE -eq 3 ]; then
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
            python3 DCTraCSprocessing.py "tr${i}_r0_ts${j}_r${i}" &&
            python3 DCTraCSresults.py "tr${i}_r0_ts${j}_r${i}" > $RESULT_FILE &&

            #rm -r ${SCK_PATH}
            echo $RESULT_FILE
        done;
    done;

elif [ $MODE -eq 4 ]; then
    for i in "${TRAIN_SIZE[@]}"; do
        if [ -d extracted_${i} ]; then
            rm -r extracted_${i}
        fi;
        mkdir extracted_${i}

        python3 generate_definitions.py "["${i}"]" "[0]" "[32]" "[0]" ${NUM_IMGS} ${PARSER} ${SCK_PATH} ${STR_CLASSES} ${DATASET} &&
        python3 DCTraCSprocessing.py "tr${i}_r0_ts32_r0" &&

        for j in "${EXTRACTORS[@]}"; do
            mkdir extracted_${i}/${j}
            cp ${SCK_PATH}${j}/class1.sck extracted_${i}/${j}/ft.sck
            cp ${SCK_PATH}${j}/class2.sck extracted_${i}/${j}/lu.sck
            cp ${SCK_PATH}${j}/class3.sck extracted_${i}/${j}/map.sck
            cp ${SCK_PATH}${j}/class4.sck extracted_${i}/${j}/allreduce.sck
        done;
    done;
fi;

