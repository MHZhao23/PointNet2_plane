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
Run the following command to train a plane segmentation model with real scene
```
python train_semseg.py --model pointnet2_sem_seg --train_path "./data_scene/crop_data" --test_path "./data_scene/crop_testdata" --batch_size 128 --epoch 64 --log_dir pointnet2_real_data --transfer
```

If the model is trained from scratch, do not add argument ```--transfer```, for example
```
python train_semseg.py --model pointnet2_sem_seg --train_path "./data_scene/crop_data" --test_path "./data_scene/crop_testdata" --batch_size 128 --epoch 64 --log_dir pointnet2_real_data

```

## Testing
Run the following command to test the model with the labelled whole scene
```
python test_labelled.py --log_dir pointnet2_real_data --test_path "./data_scene/crop_testdata" --visual --model_epoch 63
```

Run the following command to test the model with the unlabelled whole scene
```
python draw_results.py --log_dir pointnet2_real_data_0823_xyz_w4 --target_dir crop_testdata --prediction_dir eval_labelled_63

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

