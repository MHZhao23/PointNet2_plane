# Transfer pointnet2 to binary classification for plane and non-plane.

## Requirements
* Python 3.7
* Pytorch 1.13.1
* os
* sys
* argparse
* logging
* importlib
* datetime
* shutil
* pathlib
* tqdm
* numpy

## Installation
```
git clone https://github.com/xiaotaiyangzmh/PointNet2_plane.git
cd PointNet2_plane/
```

## Training
If the training uses transfer learning
Run the following command to train a plane segmentation model with plane and non-plane synthetic data (dataset 1 & dataset 2)
```
python train_synthetic_data.py --model pointnet2_sem_seg --datapath1 "./data_synthetic/pcd_plane" --datapath2 "./data_synthetic/pcd_nonplane" --epoch 32 --log_dir pointnet2_synthetic_data

```
Run the following command to train a plane segmentation model with combined synthetic data (dataset 1 & dataset 2)
```
python train_synthetic_combined_data.py  --model pointnet2_sem_seg --train_path "./data_synthetic/pcd_combined_train" --test_path "./data_synthetic/pcd_combined_test" --epoch 32 --log_dir pointnet2_synthetic_combined_data
```
Run the following command to train a plane segmentation model with real scene
```
python train_real_data.py --model pointnet2_sem_seg --datapath "./data_scene/manual_data" --epoch 32 --log_dir pointnet2_real_data

python train_test_real_data.py --model pointnet2_sem_seg --train_path "./data_scene/manual_data" --test_path "./data_scene/manual_testdata" --epoch 64 --log_dir pointnet2_real_data

python train_test_real_data.py --model pointnet2_sem_seg --train_path "./data_scene/crop_data" --test_path "./data_scene/crop_testdata" --batch_size 128 --epoch 64 --log_dir pointnet2_real_data
```

If the model is trained from scratch, add one argument ```--transfer```, for example
```
python train_synthetic_data.py --model pointnet2_sem_seg --datapath1 "./data_synthetic/pcd_plane" --datapath2 "./data_synthetic/pcd_nonplane" --epoch 32 --log_dir pointnet2_synthetic_data --transfer

```

## Testing
Run the following command to test the model with the labelled whole scene
```
python test_labelled.py --log_dir pointnet2_real_data --test_path "./data_scene/manual_testdata" --visual --model_epoch 30
python test_labelled.py --log_dir pointnet2_synthetic_data --visual --model_epoch 30
python test_labelled.py --log_dir pointnet2_synthetic_combined_data --visual --model_epoch 30
```

Run the following command to test the model with the unlabelled whole scene
```
python test_unlabelled.py --log_dir pointnet2_real_data --test_path "./data_scene/manual_testdata" --model_epoch 63
python test_unlabelled.py --log_dir pointnet2_synthetic_data --model_epoch 30
python test_unlabelled.py --log_dir pointnet2_synthetic_combined_data --model_epoch 30
```

## Testing
Run the following command to plot figures
'''
python draw_results.py --exp_dir plane_seg --log_dir pointnet2_real_data
'''

## Exporting to c++
python export_pytorch_jit.py --log_dir pointnet2_synthetic_data
python export_pytorch_jit.py --log_dir pointnet2_synthetic_combined_data

## Reference
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)<br>
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

