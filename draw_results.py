import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

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
    
    plt.plot(list(range(1, len(train_loss_list) + 1)), train_loss_list, 'r--',label='train mean loss')
    plt.plot(list(range(1, len(train_loss_list) + 1)), test_loss_list, 'b--',label='test mean loss')
    plt.title('Mean Loss during Training and Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt_savepath = save_dir.joinpath(f"Mean Loss")
    plt.savefig(plt_savepath)
    plt.show()

    plt.plot(list(range(1, len(train_loss_list) + 1)), train_acc_list, 'g--',label='train accuracy')
    plt.plot(list(range(1, len(train_loss_list) + 1)), test_acc_list, 'y--',label='test accuracy')
    plt.title('Accuracy during Training and Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt_savepath = save_dir.joinpath(f"Accuracy")
    plt.savefig(plt_savepath)
    plt.show()

    plt.plot(list(range(1, len(train_loss_list) + 1)), test_iou_list)
    plt.title('Average class IoU during Testing')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt_savepath = save_dir.joinpath(f"IOU")
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
    plt.title('Confusion matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt_savepath = save_dir.joinpath(f"confusion_matrix_{epoch}")
    plt.savefig(plt_savepath)
    plt.show()

    # array = [[TP / (total_num * 1.0), FN / (total_num * 1.0)],
    #          [FP / (total_num * 1.0), TN / (total_num * 1.0)]]

    # df_cm = pd.DataFrame(array, range(1, -1, -1), range(1, -1, -1))
    # # plt.figure(figsize=(10,7))
    # sn.set(font_scale=1.4) # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    # plt.show()
