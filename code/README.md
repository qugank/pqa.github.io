# Training Manual

## download dataset and weight

please see https://qugank.github.io/pqa.github.io

## setup environment 

setup environment via conda

`conda env create -f environment.yaml`
## training 

1. edit`script/ContextAttention/training.py`, setting`batchSize`and`gpunum`

2. in the same file, setting`dataset_folder` to the root folder of dataset, recommend to use absolute path.

3. enter the parent directory of `script`

4. running `python script/ContextAttention/training.py`
