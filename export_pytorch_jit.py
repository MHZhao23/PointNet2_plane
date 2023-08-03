import os
import sys
import argparse
import logging
import importlib
import datetime
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import SyntheticData
from models import pointnet2_sem_seg_c
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_c', help='Model name [default: pointnet2_sem_seg]')
    parser.add_argument('--transfer', action="store_false", help='Do transfer learning or not [default: True]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')

    return parser.parse_args()

def main(args):
    def log_string(args_str):
        logger.info(args_str)
        print(args_str)

    '''CHANGING DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir = experiment_dir.joinpath('sem_seg')
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    log_string('PARAMETER ...')
    log_string(args)

    '''HYPER PARAMETER'''
    num_classes = 2
    num_points = args.npoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_string("using {} device.".format(device))

    '''MODEL LOADING'''
    log_string('\n\n>>>>>>>> MODEL LOADING <<<<<<<<')
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(num_classes).to(device)
    classifier.apply(inplace_relu)

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    start_epoch = checkpoint['epoch']
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string('Use pretrain model')

    '''Export the model'''
    log_string('\n\n>>>>>>>> EXPORT MODEL <<<<<<<<')
    save_path = str(checkpoints_dir) + '/exported_model.pth'
    classifier = classifier.eval()
    points = torch.randn(16, 6, num_points).to(device)
    traced_cell = torch.jit.trace(classifier, points)
    traced_cell.save(save_path)
    print(f"model saved in {save_path}")

if __name__ == '__main__':
    args = parse_args()
    main(args)