#!/usr/bin/env bash

a=${1}
b=${2}
c=${3}
d=${4}
e=${5}
f=${6}
g=${7}
h=${8}




source ~/.bashrc
conda activate testenv3
python main.py --lr ${a} --quad_reg ${b}  --num_examples_train ${c}   --clip_grad_norm ${d}   --generative_model ${e}   --noise ${f}  --align ${g}  --align ${h}   >   /home/jkotary/QAP-LP/src/qap/log/main_runs_${h}.log
