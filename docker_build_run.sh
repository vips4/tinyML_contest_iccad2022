docker build -t tinyml-univr .

## DEBUG
# docker run --gpus all --rm --privileged -it -e PYTHONPATH=/code  -v `pwd`/checkpoints:/code/checkpoints -v `pwd`/data:/code/data tinyml-univr /bin/bash

## 1.TRAIN MODEL
# docker run --gpus all --rm --privileged -e PYTHONPATH=/code -v `pwd`/checkpoints:/code/checkpoints -v `pwd`/data:/code/data tinyml-univr python train.py --epochs 160 --autoconvert --not_concatenate_ts
## 2.EVALUATE MODEL
# docker run --gpus all --rm --privileged -e PYTHONPATH=/code -v `pwd`/checkpoints:/code/checkpoints -v `pwd`/data:/code/data tinyml-univr python evaluate.py

## FULL PIPELINE
docker run --gpus all --privileged --rm -e PYTHONPATH=/code -v `pwd`/checkpoints:/code/checkpoints -v `pwd`/data:/code/data tinyml-univr ./utils/pipeline.sh
