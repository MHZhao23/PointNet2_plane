import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Draw results')
    parser.add_argument('--exp_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--log_dir', type=str, required=True, help='log root')
    args = parser.parse_args()

    logfile_dir = 'log/' + args.exp_dir + '/' + args.log_dir + '/logs/pointnet2_sem_seg.txt'
    logfile_dir = Path(logfile_dir)
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
