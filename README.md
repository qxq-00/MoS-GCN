# General
This is the code for MoSGCN:Aspect-based sentiment analysis via distance matrix and similarity graph


# Evironment
Configure the environment required for the project.
Download spacy's en_core_web_sm language pack
```
pip install -r requirements.txt
cd MoS-GCN
python -m spacy download en_core_web_sm
```

# Preprocess Dataset
Converting the raw data to arrow format.
1. Comment out the dataset in preprocess_dataset.py that is not needed
```
python preprocess_dataset.py
```
2. The preprocessed dataset will be stored in "./dataset/arrow"

# Download Pretrained weight
## Vilt weight for pretraining
```
cd pretrained_weight
sh download_weight.sh
```
## Yoro weight for various VG tasks
1. Download result.zip from google drive [google drive](https://drive.google.com/file/d/1dqwT-YXmVdyUkPPLfm-D3hHfmCFxfk7j/view?usp=share_link)
2. unzip the result.zip


# Evaluation
For each eval.sh file in the script/DATASET, change the flag "debug" to False to run full evaluation. Below, we will describe how to run the eval.sh for different datasets.

## Pretraining tasks
```
sh script/pretrain/eval.sh
```
## Downstream tasks

### RefCoco Dataset
```
sh script/RefCoco/eval.sh
```
### RefCoco+ Dataset
```
sh script/RefCocoP/eval.sh
```
### RefCocog Dataset
```
sh script/RefCocog/eval.sh
```
### CopsRef Dataset
```
sh script/copsref/eval.sh
```
### ReferItGame/RefClef Dataset
```
sh script/ReferItGame/eval.sh
```


# Training
For all run.sh file, please change the "debug" flag to True to run the full training. 

## Pretraining tasks
For Modulated detection pretraining, we start from a mlm-itm pretrained model, such as the vilt pretraining checkpoint. For example, the below script is for training with 5 det tokens for 40 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/pretrain/run.sh 5 40 1
```

## Downstream tasks

### RefCoco Dataset
For RefCoco dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 10 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/RefCoco/run.sh 5 10 1
```
### RefCoco+ Dataset
For RefCoco+ dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 10 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/RefCocoP/run.sh 5 10 1
```
### RefCocog Dataset
For RefCocog dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 10 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/RefCocog/run.sh 5 10 1
```
### CopsRef Dataset
For copsref dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 40 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/copsref/run.sh 5 40 1
```
### ReferItGame/RefClef Dataset
For ReferItGame/RefClef dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 40 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/ReferItGame/run.sh 5 40 1
```
