# Democratic Training
This is the repository accompanying our paper TBD.

## Setup
python 3.8.10
pytorch 1.11

### Config
Copy the `sample_config.py` to `config.py` and edit the paths accordingly.

### Datasets
The code supports ImageNet, Caltech101, ASL, EuroSAT and CIFAR-10

## Run
Run `bash ./train_uap.sh` to generate targeted UAPs for different target models.

Run `bash ./repair_uap.sh` to execute democratic training to mitigate the effect of UAPs on target models.

Run `bash ./test_uap.sh` to test the generated targeted UAPs for different target models. 

## Citation
```
TBD
```
