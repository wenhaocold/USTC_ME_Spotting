#!/bin/bash

# color definition
BWHITE='\033[1;37m'
BRED='\033[1;31m'
BGREEN='\033[1;32m'
NC='\033[0m'

# enter the dataset name until it is valid.
while :
do
    echo -en "Please enter the name of dataset, the dataset should be one of ${BGREEN}cross${NC}, ${BGREEN}cas(me)^2${NC}, and ${BGREEN}samm${NC}: "
    read dataset
    if [[ "$dataset" == "cross" || "$dataset" == "cas(me)^2" || "$dataset" == "samm" ]]
    then
        break
    elif [[ "$dataset" == "q" || "$dataset" == "quit" ]]
    then
        echo "quit process $$"
        exit 0
    else
        echo -e "${BRED}Error${NC}: $dataset is invalid"
    fi
done

if [ "$dataset" == "cross" ]; then
    echo -e "${BWHITE}training and evaluati on${NC} ${BGREEN}$dataset${NC}"
    python main.py --dataset=$dataset
else
    # enter the phase until it is valid
    while :
    do
        echo -en "Please enter the phase you want, ${BGREEN}train${NC} or ${BGREEN}eval${NC}: "
        read t_or_e
        if [[ "$t_or_e" == "train" || "$t_or_e" == "eval" ]]; then
            break
        elif [[ "$t_or_e" == "q" || "$t_or_e" == "quit" ]]; then
            echo "quit process $$"
            exit 0
        else
            echo -e "${BRED}Error${NC}: $t_or_e is invalid"
        fi
    done

    if [ "$t_or_e" == "train" ]; then
        echo -e "${BWHITE}training on${NC} ${BGREEN}$dataset${NC}"
        python main.py --dataset=$dataset --mode=abfcm_train_mul_process
    elif [ "$t_or_e" == "eval" ]; then
        echo -e "${BWHITE}evaluation on${NC} ${BGREEN}$dataset${NC}"
        python main.py --dataset=$dataset --mode=abfcm_output_mul_process &&
        python main.py --dataset=$dataset --mode=abfcm_nms_mul_process &&
        python main.py --dataset=$dataset --mode=abfcm_iou_mul_process &&
        python main.py --dataset=$dataset --mode=abfcm_final_result
    else
        echo -e "${BRED}Error${NC}: invalida value"
        exit 1
    fi
fi

