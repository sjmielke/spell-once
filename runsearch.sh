#!/usr/bin/env bash

MAINDIR="../spell-once"

cd "$MAINDIR"

for data in wikitext-2-raw; do
# for data in mwc{-tok,}/wiki_{cs,de,en,es,fi,fr,ru}; do
# for vocab in 1000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000; do
# for vocab in 10000 30000 50000 70000 90000 110000 130000 150000; do
for vocab in 30000 40000 50000 60000; do
# for data in mwc-tok/wiki_{cs,de,en,es,fi,fr,ru}; do
# for vocab in 1000 10000 20000 30000 40000 50000 60000 70000 80000 90000; do
# for vocab in 10000 30000 50000 70000 90000; do
# for mode in {full,1gram,uncond}-typ backoff-tok; do
for mode in sep-backoff-typ; do
for bs in 40; do

# MODELPREFIX="tune_${data//\//_}_vocab${vocab}_${mode}_bs${bs}_epo1300"
MODELPREFIX="test_vocab${vocab}_bs${bs}_${mode}"

cat <<- EOF > "_${MODELPREFIX}.sh"
#!/usr/bin/env bash
cd "$MAINDIR"

# CUDA_VISIBLE_DEVICES=\`free-gpu\` \
# unbuffer \

python main.py \
  --cuda \
  --open-vocab \
  --open-vocab-during-training \
  --data data/${data} \
  --dropouth 0.2 \
  --batch_size ${bs} \
  --vocab-size ${vocab} \
  --speller-mode ${mode} \
  --save models/${MODELPREFIX}.pt \
  --boardcomment ${MODELPREFIX} \
  --epochs 1300 \
| tee models/${MODELPREFIX}.log
EOF

# qsub -l 'hostname=c*,gpu=1' -q g.q "_${MODELPREFIX}.sh"
sbatch -p gpu --gres=gpu:1 --mem=32G --time=100:00:00 "_${MODELPREFIX}.sh"

done
done
done
done

# BPE experiments

#for data in mwc{-tok,}-bpe/wiki_{cs,de,en,es,fi,fr,ru}; do
# for data in mwc{-tok,}/wiki_{cs,de,en,es,fi,fr,ru}-bpe-{1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150}k; do
# for data in \
#   mwc/wiki_cs-bpe-{1,10,20,30,40,50,60,70,80,90,100,110}k \
#   mwc/wiki_de-bpe-{1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150}k \
#   mwc/wiki_en-bpe-{1,10,20,30,40,50,60,70,80,90,100,110,120,130}k \
#   mwc/wiki_es-bpe-{1,10,20,30,40,50,60,70,80,90,100,110}k \
#   mwc/wiki_fi-bpe-{1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150}k \
#   mwc/wiki_fr-bpe-{1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150}k \
#   mwc/wiki_ru-bpe-{1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150}k
#   mwc-tok/wiki_cs-bpe-{1,10,20,30,40,50,60,70,80,90}k \
#   mwc-tok/wiki_de-bpe-{1,10,20,30,40,50,60,70,80,90,100,110,120}k \
#   mwc-tok/wiki_en-bpe-{1,10,20,30,40,50,60,70,80,90}k \
#   mwc-tok/wiki_es-bpe-{1,10,20,30,40,50,60,70,80}k \
#   mwc-tok/wiki_fi-bpe-{1,10,20,30,40,50,60,70,80,90,100}k \
#   mwc-tok/wiki_fr-bpe-{1,10,20,30,40,50,60,70,80}k \
#   mwc-tok/wiki_ru-bpe-{1,10,20,30,40,50,60,70,80,90,100,110,120}k; do
for data in mwc{-tok,}/wiki_{cs,de,en,es,fi,fr,ru}-bpe-50k; do

MODELPREFIX="tune_${data//\//_}"
cat <<- EOF > "_${MODELPREFIX}.sh"
#!/usr/bin/env bash
cd "$MAINDIR"

# CUDA_VISIBLE_DEVICES=\`free-gpu\` \
# unbuffer \

python main.py \
  --cuda \
  --data data/${data} \
  --dropouth 0.2 \
  --batch_size 40 \
  --vocab-size 999999 \
  --save models/${MODELPREFIX}.pt \
  --boardcomment ${MODELPREFIX} \
  --speller-mode none \
| tee models/${MODELPREFIX}.log
EOF

# sbatch -p gpu --gres=gpu:1 --mem=32G --time=100:00:00 "_${MODELPREFIX}.sh"

done
