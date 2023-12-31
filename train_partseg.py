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
import open3d as o3d

from datasets import RealData, SceneLabelledData
from models import pointnet2_sem_seg
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['non-plane', 'plane']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

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

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
    parser.add_argument('--train_path', type=str, default=None, help='Rootpath of data, "./data_scene" [default: None]')
    parser.add_argument('--test_path', type=str, default=None, help='Rootpath of data, "./data_scene" [default: None]')
    parser.add_argument('--fw', type=float, default=2.0, help='Power of labelsweight [default: 2.0]')
    parser.add_argument('--rgb', action="store_true", help='Train with RGB channels [default: False]')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--transfer', action="store_true", help='Do transfer learning or not [default: False]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=512, help='Point Number [default: 4096]')
    parser.add_argument('--block_size', type=float, default=0.2, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()

def main(args):
    def log_string(args_str):
        logger.info(args_str)
        print(args_str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('plane_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    models_cp_dir = checkpoints_dir.joinpath('models/')
    models_cp_dir.mkdir(exist_ok=True)
    bestmodels_cp_dir = checkpoints_dir.joinpath('best_models/')
    bestmodels_cp_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''HYPER PARAMETER'''
    num_classes = 2
    num_points = args.npoint
    block_size = args.block_size
    batch_size = args.batch_size
    lr_net = args.learning_rate
    epochs = args.epoch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_string("using {} device.".format(device))

    '''DATASET LOADING'''
    log_string('\n\n>>>>>>>> DATASET LOADING <<<<<<<<')
    log_string("start loading training data ...")
    train_set = RealData(args.train_path, num_classes, num_points, block_size, fw=args.fw, rgb=args.rgb)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True, 
                              drop_last=True,
                              num_workers=0)
    log_string("using {} samples for training.".format(train_set.__len__()))
    weights = torch.Tensor(train_set.labelweights).to(device)

    log_string("\nstart loading testing data ...")
    test_set = RealData(args.test_path, num_classes, num_points, block_size, fw=args.fw, rgb=args.rgb)
    test_loader = DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True, 
                              drop_last=True,
                              num_workers=0)
    log_string("using {} samples for testing.".format(test_set.__len__()))

    '''MODEL LOADING'''
    log_string('\n\n>>>>>>>> MODEL LOADING <<<<<<<<')
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    if args.rgb:
        in_channels = 6
    else:
        in_channels = 3
    classifier = MODEL.get_model(in_channels, num_classes).to(device)
    criterion = MODEL.get_loss().to(device)
    classifier.apply(inplace_relu)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        start_epoch = 0
        classifier = classifier.apply(weights_init)
        if args.transfer == True:
            log_string('No existing model, using transfer learning...')
            transfer_model = torch.load('./weights/partseg_msg_model.pth')
            classifier_state_dict = classifier.state_dict()
            for key in transfer_model['model_state_dict'].keys():
                if key in classifier_state_dict.keys():
                    if transfer_model['model_state_dict'][key].shape == classifier_state_dict[key].shape:
                        classifier_state_dict[key] = transfer_model['model_state_dict'][key]
        else:
            log_string('No existing model, starting training from scratch...')

    if args.transfer == True:
        require_grad_layer = ["conv0", "sa1", "conv1", "conv2"]
        for param in classifier.named_parameters():
            if param[0][:3] in require_grad_layer or param[0][:5] in require_grad_layer:
                param[1].requires_grad = True
            else:
                param[1].requires_grad = False
        grad_params = [p for p in classifier.parameters() if p.requires_grad]
    else:
        grad_params = [p for p in classifier.parameters()]

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(grad_params,
            lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(grad_params, lr=args.learning_rate, momentum=0.9)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_acc = 0
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], []

    '''Train on chopped scenes'''
    log_string('\n\n>>>>>>>> TRAINING & TESTING <<<<<<<<')
    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(train_loader)
        total_correct = 0
        total_seen = 0
        total_plane = 0
        total_nonplane = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.random_scale_point_cloud(points[:, :, :3])
            points[:, :, :3] = provider.shift_point_cloud(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points, to_categorical(target, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_classes)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            plane_num = np.sum(pred_choice == 1)
            nonplane_num = np.sum(pred_choice == 0)
            total_plane += plane_num
            total_nonplane += nonplane_num
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (batch_size * num_points)
            loss_sum += loss
        log_string('Plane: %d, %f; Non-plane: %d, %f;' % (total_plane, total_plane/(total_plane+total_nonplane), total_nonplane, total_nonplane/(total_plane+total_nonplane)))
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
        train_loss_list.append((loss_sum / num_batches).cpu().detach().numpy())
        train_acc_list.append(total_correct / float(total_seen) * 100)

        if epoch % 3 == 0:
            log_string('Saving model....')
            savepath = str(models_cp_dir) + f'/model_{epoch}.pth'
            log_string('Model Saved at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Model saved successfully!')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(test_loader)
            total_correct = 0
            total_seen = 0
            total_plane = 0
            total_nonplane = 0
            loss_sum = 0
            labelweights = np.zeros(num_classes)
            total_seen_class = [0 for _ in range(num_classes)]
            total_correct_class = [0 for _ in range(num_classes)]
            total_iou_deno_class = [0 for _ in range(num_classes)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().to(device), target.long().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, num_classes)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                plane_num = np.sum(pred_val == 1)
                nonplane_num = np.sum(pred_val == 0)
                total_plane += plane_num
                total_nonplane += nonplane_num
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (batch_size * num_points)
                tmp, _ = np.histogram(batch_label, range(num_classes + 1))
                labelweights += tmp

                for l in range(num_classes):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            class_IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
            mIoU = np.sum(labelweights * class_IoU)
            class_acc = np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6)
            macc = np.sum(labelweights * class_acc)
            avgacc = total_correct / float(total_seen)
            log_string('Plane: %d, %f; Non-plane: %d, %f;' % (total_plane, total_plane/(total_plane+total_nonplane), total_nonplane, total_nonplane/(total_plane+total_nonplane)))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (avgacc))
            log_string('eval point avg class acc: %f' % (macc))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(num_classes):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
            test_loss_list.append((loss_sum / num_batches).cpu().detach().numpy())
            test_acc_list.append(total_correct / float(total_seen) * 100)

            if avgacc >= best_acc:
                best_acc = avgacc
                log_string('Saving model....')
                savepath = str(bestmodels_cp_dir) + f'/best_model_{epoch}.pth'
                log_string('Model Saved at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_acc': avgacc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Best model saved successfully!')

            elif epoch == (args.epoch - 1):
                log_string('Saving model....')
                savepath = str(bestmodels_cp_dir) + f'/best_model_{epoch}.pth'
                log_string('Model Saved at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_acc': avgacc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Best model saved successfully!')

            log_string('Best accuracy: %f' % best_acc)

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)



