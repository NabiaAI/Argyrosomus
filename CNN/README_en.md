

## Installation Environment
   You can check requirement.txt


## Preparing Data

`create_data.py` can be used to generate a list of data sets. You can change whatever you want.

```shell
python create_data.py
```

## Extract features
```shell
python extract_features.py
```

You can change the parameters in yaml.

Then you can get the feature.txt in the path  path `data/`



## Training

Change the yaml parameter (train_list and test_list )

  dataset_conf:

    train_list: 'data/train0_features.txt'
    test_list: 'data/test0_features.txt'
    label_list_path: 'data/label_list.txt'

```shell
# Single GPU training
CUDA_VISIBLE_DEVICES=0 python train.py
# Multi GPU training
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```


# Eval
```shell
python eval.py
```


