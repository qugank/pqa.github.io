# Training Manual

## download dataset and trained model

please refer to <a href="https://qugank.github.io/pqa.github.io" target="_blank">project page<a> to download the dataset.

Trained model is available <a href="https://drive.google.com/file/d/1FW2SMdd68U2KpSxTG4QEYQfexVn1KjK2/view?usp=sharing" target="_blank">here<a>.
  
## setup environment 

setup environment via conda

`conda env create -f environment.yaml`
## training 

1. edit`script/ContextAttention/training.py`, setting`batchSize`and`gpunum`

2. in the same file, setting`dataset_folder` to the root folder of dataset, recommend to use absolute path.

3. enter the parent directory of `script`

4. running `python script/ContextAttention/training.py`
