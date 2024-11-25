# Democratic Training
This is the repository accompanying our paper TBD.

## Setup
python 3.8.10
pytorch 1.11

### Config
Copy the `sample_config.py` to `config.py` and edit the paths accordingly.

### Datasets
The code supports ImageNet, Caltech101, ASL, EuroSAT and CIFAR-10

#### ImageNet
    - uncompress imagenet validation set to IMAGENET_PATH/validation/val
    - in IMAGENET_PATH/validation directory, do below:
        * copy classes.txt to IMAGENET_PATH/validation
        * run pre_process_rename.sh
        * run generate_test_index.py to generate 2000 random indexes for testing

#### CIFAR-10
Dataset will be downloaded automatically.

## Run
Run `bash ./train_uap.sh` to generate targeted UAPs for different target models.

Run `bash ./repair_uap.sh` to execute democratic training to mitigate the effect of UAPs on target models.

Run `bash ./test_uap.sh` to test the generated targeted UAPs for different target models. 

## Pre-trained Models for Testing

[cifar10-wideresnet](https://figshare.com/s/51d4ce8b1ab552e57bd2)

## Citation
```
TBD
```
