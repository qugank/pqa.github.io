# Perceptual Question Answering

## environment 

setup environment via conda

`conda env create -f environment.yaml`

## download dataset

1. old version : https://drive.google.com/file/d/1Ump44358mfhziPMeEpdDYJHEUne9wcLl/view?usp=sharing

2. latest version (recommended) : https://drive.google.com/file/d/1OemRMUogRpvQ257de9JA_4NZJwRLyMLb/view?usp=sharing


## weight

1. weight：https://drive.google.com/file/d/1FW2SMdd68U2KpSxTG4QEYQfexVn1KjK2/view?usp=sharing

2. inti weight：https://drive.google.com/file/d/18g9LXyhcyHfC-2V_BPPQpXk8FztDkXe7/view?usp=sharing

## training 

1. edit`script/ContextAttention/training.py`, setting`batchSize`and`gpunum`

2. in the same file, setting`dataset_folder` to the root folder of dataset, recommend to use absolute path.

3. enter the parent directory of `script`

4. running `python script/ContextAttention/training.py`
