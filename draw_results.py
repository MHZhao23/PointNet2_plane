import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import open3d as o3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Draw results')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
    parser.add_argument('--exp_dir', type=str, default='plane_seg', help='experiment root')
    parser.add_argument('--log_dir', type=str, required=True, help='log root')
    parser.add_argument('--target_dir', type=str, required=True, help='target data root')
    parser.add_argument('--prediction_dir', type=str, required=True, help='prediction data root')
    args = parser.parse_args()

    logfile_dir = 'log/' + args.exp_dir + '/' + args.log_dir + '/logs/' + args.model + ".txt"
    logfile_dir = Path(logfile_dir)
    target_dir = './data_scene/' + args.target_dir + '/label'
    prediction_dir = './log/' + args.exp_dir + '/' + args.log_dir + '/' + args.prediction_dir
    epoch = prediction_dir.split("_")[-1]
    print(target_dir)
    print(prediction_dir)
    save_dir = 'log/' + args.exp_dir + '/' + args.log_dir + '/plots'
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    test_iou_list = []
    with open(logfile_dir) as f:
        lines = f.readlines()
    for l in lines:
        if "Training mean loss" in l:
            l = l.split()
            train_loss_list.append(float(l[-1]))
        elif "Training accuracy" in l:
            l = l.split()
            train_acc_list.append(float(l[-1]) * 100)
        elif "eval mean loss" in l:
            l = l.split()
            test_loss_list.append(float(l[-1]))
        elif "eval point accuracy" in l:
            l = l.split()
            test_acc_list.append(float(l[-1]) * 100)
        elif "eval point avg class IoU" in l:
            l = l.split()
            test_iou_list.append(float(l[-1]))
    assert len(train_loss_list) == len(test_loss_list) == len(train_acc_list) == len(test_acc_list) ==len(test_iou_list)
    
    plt.plot(list(range(1, len(train_loss_list) + 1)), train_loss_list, 'r--',label='Loss on the training set')
    plt.plot(list(range(1, len(train_loss_list) + 1)), test_loss_list, 'b--',label='Loss on the testing set')
    plt.title('The Negative Log Likelihood Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt_savepath = save_dir.joinpath(f"Loss")
    plt.savefig(plt_savepath)
    plt.show()

    plt.plot(list(range(1, len(train_loss_list) + 1)), train_acc_list, 'g--',label='Accuracy on the training set')
    plt.plot(list(range(1, len(train_loss_list) + 1)), test_acc_list, 'y--',label='Accuracy on the testing set')
    plt.title('Accuracy during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt_savepath = save_dir.joinpath(f"Accuracy")
    plt.savefig(plt_savepath)
    plt.show()

    target_filenames = sorted(os.listdir(target_dir))
    target_files = [os.path.join(target_dir, filename) for filename in target_filenames if filename.endswith("npy")]

    prediction_filenames = sorted(os.listdir(prediction_dir))
    prediction_files = [os.path.join(prediction_dir, filename) for filename in prediction_filenames if filename.endswith("npy")]
    assert len(target_files) == len(prediction_files)
    
    total_num = 0
    TP, FN, FP, TN = 0, 0, 0, 0

    for i in range(len(target_files)):
        labels = np.load(target_files[i])
        preds = np.load(prediction_files[i])

        TP += np.sum((labels == 1) & (preds == 1))
        FN += np.sum((labels == 1) & (preds == 0))
        FP += np.sum((labels == 0) & (preds == 1))
        TN += np.sum((labels == 0) & (preds == 0))
        total_num += labels.shape[0]

    cf = np.array([[round((TN / (total_num * 1.0)), 3), round((FP / (total_num * 1.0)), 3)], 
                   [round((FN / (total_num * 1.0)), 3), round((TP / (total_num * 1.0)), 3)]])
    fig, ax = plt.subplots()
    ax.matshow(cf, cmap=plt.cm.Blues)
    for i in range(2):
        for j in range(2):
            c = cf[i,j]
            ax.text(j, i, str(c), va='center', ha='center')
    plt.title('Confusion Matrix on the Testing set')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt_savepath = save_dir.joinpath("confusion_matrix")
    plt.savefig(plt_savepath)
    plt.show()

    root_path = f"/home/minghan/workspace/plane_detection_NN/PointNet2_plane/log/{args.exp_dir}/{args.log_dir}/{args.prediction_dir}"
    file_list = sorted(os.listdir(root_path))
    gtpcd_file = [os.path.join(root_path, f) for f in file_list if f.endswith("gt.pcd")]
    predpcd_file = [os.path.join(root_path, f) for f in file_list if f.endswith("pred.pcd")]
    for i in range(len(predpcd_file)):
        pcd_f = gtpcd_file[i]
        pcd = o3d.io.read_point_cloud(pcd_f)
        o3d.visualization.draw_geometries([pcd])

        pcd_f = predpcd_file[i]
        pcd = o3d.io.read_point_cloud(pcd_f)
        o3d.visualization.draw_geometries([pcd])
