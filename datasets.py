import os
import re
import open3d as o3d
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class RealData(Dataset):
    """
    This class create dataset with both planes and non-planes.
    
    Arguments:

    rootpath: root path of data, "./data_scene"
    num_classes: the number of classes
    num_point: the number of points in each sampling group
    train_ratio: the ratio of train and test
    mod: train or test
    """
    
    def __init__(self, rootpath, num_classes, num_point, block_size, fw, rgb=False):

        super(RealData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size
        self.rgb = rgb

        # loading data
        if self.rgb:
            cloud_path = os.path.join(rootpath, "rgb_cloud")
        else:
            cloud_path = os.path.join(rootpath, "cloud")
        cloud_filename = sorted(os.listdir(cloud_path))
        cloud_files = [os.path.join(cloud_path, filename) for filename in cloud_filename]

        label_path = os.path.join(rootpath, "label")
        label_filename = sorted(os.listdir(label_path))
        label_files = [os.path.join(label_path, filename) for filename in label_filename]
        assert len(cloud_files) == len(label_files)

        model_num = len(cloud_files)
        indices = list(range(model_num))
        print(f"loading {len(indices)} models ...")

        self.points_list, self.labels_list = [], []
        self.coord_min_list, self.coord_max_list = [], []
        num_point_all = []
        labelweights = np.zeros(num_classes)

        for i in tqdm(indices, total=len(indices)):
            # load point cloud
            pcd = o3d.io.read_point_cloud(cloud_files[i])
            points = np.asarray(pcd.points)
            points = points - np.amin(points, axis=0)[:3]
            if self.rgb:
                colors = np.asarray(pcd.colors) / 255.0
                points = np.concatenate((points, colors), axis=1)

            # load labels
            labels = np.load(label_files[i]).astype(np.float64)
            tmp, _ = np.histogram(labels, range(3))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.points_list.append(points), self.labels_list.append(labels)
            self.coord_min_list.append(coord_min), self.coord_max_list.append(coord_max)
            num_point_all.append(labels.size)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, fw)
        # self.labelweights = labelweights / np.sum(labelweights)

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) / num_point)

        cloud_idxs = []
        for index in range(len(indices)):
            cloud_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.cloud_idxs = np.array(cloud_idxs)

        assert len(self.points_list) == len(self.labels_list)
        print(f"loading {len(self.points_list)} models successfully!")
        print(f"Totally {len(self.cloud_idxs)} samples")
            
    def __getitem__(self, idx):
        cloud_idx = self.cloud_idxs[idx]
        points = self.points_list[cloud_idx]
        labels = self.labels_list[cloud_idx]
        N_points = points.shape[0]

        # find the sampling center
        iter_num = 0
        tmp_size = self.block_size
        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [tmp_size / 2.0, tmp_size / 2.0, 0]
            block_max = center + [tmp_size / 2.0, tmp_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            # print(point_idxs.size, iter_num)
            if point_idxs.size > 256:
                break
            else:
                iter_num += 1

            # increase the block scale if the center cannot be found
            if iter_num % 5 == 0:
                tmp_size += self.block_size/2

        # print(points.shape, point_idxs.size, tmp_size)
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize the sampled points
        selected_points = points[selected_point_idxs, :]  # num_point * 3 or num_point * 6
        if self.rgb:
            current_points = np.zeros((self.num_point, 9))
            current_points[:, :3] = selected_points[:, :3] - center
            current_points[:, 3:6] = selected_points[:, 3:6]
            selected_points[:, :3] = selected_points[:, :3] - np.amin(selected_points[:, :3], axis=0)
            current_points[:, 6:] = selected_points[:, :3] / np.amax(selected_points[:, :3], axis=0)
        else:
            current_points = np.zeros((self.num_point, 6))
            current_points[:, :3] = selected_points[:, :3] - center
            selected_points = selected_points - np.amin(selected_points, axis=0)
            current_points[:, 3:] = selected_points / np.amax(selected_points, axis=0)
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

    def __len__(self):

        return len(self.cloud_idxs)


class SceneLabelledData():
    """
    This class create dataset with the whole scenes.
    
    Arguments:

    rootpath: root path of data, "./data_scene"
    num_classes: the number of classes
    num_point: the number of points in each sampling group
    train_ratio: the ratio of train and test
    mod: train or test
    """
    
    def __init__(self, rootpath, num_classes, num_point, block_size, fw, padding=0.001, rgb=False):
        super(SceneLabelledData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size
        self.padding = padding
        self.rootpath = rootpath
        self.stride = block_size / 2
        self.rgb = rgb

        # loading data
        if self.rgb:
            cloud_path = os.path.join(rootpath, "rgb_cloud")
        else:
            cloud_path = os.path.join(rootpath, "cloud")
        self.cloud_filename = sorted(os.listdir(cloud_path))
        cloud_files = [os.path.join(cloud_path, filename) for filename in self.cloud_filename]

        label_path = os.path.join(rootpath, "label")
        label_filename = sorted(os.listdir(label_path))
        label_files = [os.path.join(label_path, filename) for filename in label_filename]
        assert len(cloud_files) == len(label_files)

        model_num = len(cloud_files)
        self.indices = list(range(model_num))
        print(f"loading {len(self.indices)} models ...")

        self.scene_points_num = []
        self.scene_points_list = []
        self.semantic_labels_list = []
        labelweights = np.zeros(num_classes)

        for i in tqdm(self.indices, total=len(self.indices)):
            # load point cloud
            pcd = o3d.io.read_point_cloud(cloud_files[i])
            points = np.asarray(pcd.points)
            points = points - np.amin(points, axis=0)[:3]
            if self.rgb:
                colors = np.asarray(pcd.colors) / 255.0
                points = np.concatenate((points, colors), axis=1)

            # load labels
            labels = np.load(label_files[i]).astype(np.float64)
            tmp, _ = np.histogram(labels, range(num_classes+1))
            labelweights += tmp

            self.scene_points_num.append(points.shape[0])
            self.scene_points_list.append(points)
            self.semantic_labels_list.append(labels)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, fw)

    def __getitem__(self, index):
        points = self.scene_points_list[index]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_scene, label_scene, sample_weight, index_scene = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                # ignore those isolated points (the labels of those points are 0)
                if point_idxs.size < 5:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.num_point))
                point_size = int(num_batch * self.num_point)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 2] = data_batch[:, 2] - np.mean(data_batch[:, 2])
                normlized_xyz = data_batch[:, :3] - np.amin(data_batch[:, :3], axis=0)
                normlized_xyz = normlized_xyz / np.amax(normlized_xyz, axis=0)
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_scene = np.vstack([data_scene, data_batch]) if data_scene.size else data_batch
                label_scene = np.hstack([label_scene, label_batch]) if label_scene.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_scene.size else batch_weight
                index_scene = np.hstack([index_scene, point_idxs]) if index_scene.size else point_idxs
        data_scene = data_scene.reshape((-1, self.num_point, data_scene.shape[1]))
        label_scene = label_scene.reshape((-1, self.num_point))
        sample_weight = sample_weight.reshape((-1, self.num_point))
        index_scene = index_scene.reshape((-1, self.num_point))
        return data_scene, label_scene, sample_weight, index_scene

    def __len__(self):
        return len(self.scene_points_list)


class SceneUnlabelledData():
    """
    This class create dataset with the whole scenes.
    
    Arguments:

    rootpath: root path of data, "./data_scene"
    num_classes: the number of classes
    num_point: the number of points in each sampling group
    train_ratio: the ratio of train and test
    mod: train or test
    """
    
    def __init__(self, rootpath, num_classes, num_point, block_size, padding=0.001, rgb=False):
        super(SceneUnlabelledData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size
        self.padding = padding
        self.rootpath = rootpath
        self.stride = block_size / 2
        self.rgb = rgb

        # loading data
        if self.rgb:
            cloud_path = os.path.join(rootpath, "rgb_cloud")
        else:
            cloud_path = os.path.join(rootpath, "cloud")
        self.cloud_filename = sorted(os.listdir(cloud_path))
        cloud_files = [os.path.join(cloud_path, filename) for filename in self.cloud_filename]

        model_num = len(cloud_files)
        self.indices = list(range(model_num))
        print(f"loading {len(self.indices)} models ...")

        self.scene_points_num = []
        self.scene_points_list = []

        for i in tqdm(self.indices, total=len(self.indices)):
            # load point cloud
            pcd = o3d.io.read_point_cloud(cloud_files[i])
            points = np.asarray(pcd.points)
            points = points - np.amin(points, axis=0)[:3]
            if self.rgb:
                colors = np.asarray(pcd.colors) / 255.0
                points = np.concatenate((points, colors), axis=1)

            self.scene_points_num.append(points.shape[0])
            self.scene_points_list.append(points)

    def __getitem__(self, index):
        points = self.scene_points_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_scene, index_scene = np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                # ignore those isolated points (the labels of those points are 0)
                if point_idxs.size < 5:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.num_point))
                point_size = int(num_batch * self.num_point)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 2] = data_batch[:, 2] - np.mean(data_batch[:, 2])
                normlized_xyz = data_batch[:, :3] - np.amin(data_batch[:, :3], axis=0)
                normlized_xyz = normlized_xyz / np.amax(normlized_xyz, axis=0)
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)

                data_scene = np.vstack([data_scene, data_batch]) if data_scene.size else data_batch
                index_scene = np.hstack([index_scene, point_idxs]) if index_scene.size else point_idxs
        data_scene = data_scene.reshape((-1, self.num_point, data_scene.shape[1]))
        index_scene = index_scene.reshape((-1, self.num_point))
        return data_scene, index_scene

    def __len__(self):
        return len(self.scene_points_list)