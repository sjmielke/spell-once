#!/bin/bash

MAINDIR="../spell-once"

cd "$MAINDIR"

for do in 0.1 0.3; do
for lr in 5 30 50; do
for bptt in 100 500 1000; do
for bs in 20 80 200; do

STUB=char_do${do}_lr${lr}_bptt${bptt}_bs${bs}

cat <<- EOF > "_wt2_$STUB.sh"
#!/usr/bin/env bash
cd "$MAINDIR"

# CUDA_VISIBLE_DEVICES=\`free-gpu\` \
# unbuffer \

python main.py \
  --cuda \
  --epochs 750 \
  --data data/wikitext-2-raw-char \
  --save models/WT2_pure${STUB}.pt \
  --seed 42 \
  --speller-mode none \
  --bptt ${bptt} \
  --emsize 10 \
  --dropout ${do} \
  --dropoute ${do} \
  --dropouth ${do} \
  --dropouti ${do}\
  --batch_size ${bs} \
  --lr-lm ${lr} \
  --boardcomment ${STUB} \
  | tee models/WT2_pure${STUB}.log
EOF

# qsub -l 'hostname=c*,gpu=1' -q g.q "$STUB.sh"
sbatch -p gpu --gres=gpu:1 --mem=32G --time=72:00:00 "_wt2_$STUB.sh"

done
done
done
done
