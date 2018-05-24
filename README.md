# futurelab-ai-2018-image

The code for [FutureLab.AI 2018 scene image classification competition](http://ai.futurelab.tv/view). Final score on TestB: 0.99186.

## 0. Requirements

- Ubuntu 16.04 or Ubuntu 14.04
- CUDA 8.0
- cuDNN 6.0
- Python 3.6 (Miniconda 3)

Use the following command to install Python 3 dependices:

```sh
pip install -r requirements.txt
```

## 1. Training-validation splitting

```sh
sh scripts/split_train_validation.sh
```

The script will split the `training-list-0511.csv` into 11 folds and then combined them into 11 different train-val splitting, like `11-fold-x-train.csv` and `11-fold-x-val.csv` in `train_data_lists` folder. For each splitting, the ratio of training set and validation set is 10:1. If time is enough, a full cross-validation should be employed. However, we only used `11-fold-1-train.csv` and `11-fold-1-val.csv` actually.

## 2. Train

Based on the pretrained ImageNet models in [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch), we trained `inceptionv4` and `se_resnext101_32x4d` with PyTorch 0.3.1. Thanks for @Cadene's such a great repo, all pretrained models can be downloaded automatically when training.

```sh
sh scripts/train_inceptionv4.sh

sh scripts/train_se_resnext101_32x4d.sh
```

The `data_root` arguments in the above two training scripts should be designated to the training data root, which contains all the training images.

After training, the models with best top-3 accuracy performance will be saved to corresponding directory under `checkpoints`.

## 3. Test

### (1) Download the models

Download the models we trained from Baidu Netdisk: https://pan.baidu.com/s/1vnUBGrOAaBoiZmCfLxLeSg. Extracting code: `8m5h`.

After downloading, extract the zip file to `checkpoints` folder. The structure of `checkpoints` you should see:

```text
checkpoints/
|-- inceptionv4_11-fold-1
|   `-- model_best.pth.tar
`-- se_resnext101_32x4d_11-fold-1
    `-- model_best.pth.tar
```

### (2) Test the models

When testing, at first we resize the test image into a designated size, then we employ [10-crop](https://pytorch.org/docs/master/torchvision/transforms.html#torchvision.transforms.TenCrop) to get a robust test output.

For the two models we trained, we averaged the test logits they output and got the final top-3 predictions.

```sh
sh scripts/test_inceptionv4_logits.sh

sh scripts/test_se_resnext101_32x4d_logits.sh
```

Similarly to training, the `test_data_root` argument should be designated with the root of images in TestB dataset.

The logits will be saved to corresponding directories under `checkpoints`, like the follwing structure:

```text
checkpoints/
|-- inceptionv4_11-fold-1
|   |-- model_best.pth.tar
|   `-- test_logits.csv
`-- se_resnext101_32x4d_11-fold-1
    |-- model_best.pth.tar
    `-- test_logits.csv
```

### (3) Average the logits

```text
sh scripts/ensemble_logits.sh
```

Running this script will get two `csv` files under `ensemble_results` folder, which correspond to the averaged logits and the predicted top-3 labels respectively:

```text
ensemble_results/
|-- logits_ensemble.csv
`-- test_pred_ensemble.csv
```

The `test_pred_ensemble.csv` is the result what we submit to the competition. This version of models got a 0.99186 score on TestB dataset.