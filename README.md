# DEGNet: Decoupled Edge Guidance Network for Automatic Check-out

This repository implements DEGNet on the base of **Pytorch**. The implementation code of our **Cascade RCNN** framework references **<a href="https://github.com/open-mmlab/mmdetection/tree/3.x" target="_blank">MMDetection</a>**.


## Installation

Check [requirements.txt](requirements.txt) for installation instructions. Similar torch versions should also work normally.

## Data Preparation

The RPC dataset can be download by these two urls:   
    https://www.kaggle.com/diyer22/retail-product-checkout-dataset  
    https://pan.baidu.com/s/1vrrLaSpJe5JxT3zhYfOaog 
Besides, you can use the method of [DPSNet](https://github.com/jianzhnie/DPSNet/tree/master/dpsnet) to generate synthesized images.

## Necessary modifications

Before training, you should modify the contents of the paths in the following files.
1. train_utils/rpc/paths_catalog.py  
    In these configuration files, you need modify the following parameters.  
    ```
    rpc_train_syn: '{your synthesized data root path and annotation path}'
    rpc_instance: '{your rpc_validation_dataset root path and annotation path}'
    rpc_2019_test: '{your rpc_test_dataset root path and annotation path}'  
    ```
2. configs/xxx.py
    The configuration files we provide run on a 3090 GPU by default. You can adjust the global batch size and learning rate according to your graphics card configuration.
    ```
    batch_size: '{your batch_size}'
    lr: '{your learning rate}' 
    ```
## Train

We train our model in one 3090 card, and the training command is:  

    python tools/train.py --config {config file} 
and an example is:  

    python tools/train.py --config configs/ffm_syn_att_da_pds.yaml 
  

## Delete a file

You can delete the current file by clicking the **Remove** button in the file explorer. The file will be moved into the **Trash** folder and automatically deleted after 7 days of inactivity.


## Test

If you only want to evaluate the model, you can execute:

     python tools/test.py --config {config file} --cpt {checkpoint_file_path}  
and an example is:  

    python tools/test.py --config configs/ffm_syn_att_da_pds.yaml  \
           --cpt result/ffm_syn_att_da_pds/latest.pt
           

## Citation
Please consider citing our repo if it helps your research.
