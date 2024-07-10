#!/bin/bash

for i in $(seq 2 2 28); do
    for t in $(seq 16 2 20); do
        python3 test_acc.py --run_name mnistheatmap --img_size $i --task_exp $t
    done
done