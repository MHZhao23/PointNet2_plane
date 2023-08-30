import os
import sys
import argparse
import logging
import importlib
import datetime
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d


if __name__ == "__main__":

    # ## -------------- show pcd --------------
    # folder = "unlabelled"
    # epoch = 63
    # exp_folder = "pointnet2_real_data_0815"
    # root_folder = f"eval_{folder}_{epoch}"
    # root_path = os.path.join(f"/home/minghan/workspace/plane_detection_NN/PointNet2_plane/log/plane_seg/{exp_folder}/", root_folder)
    # file_list = sorted(os.listdir(root_path))
    # predpcd_file = [os.path.join(root_path, f) for f in file_list if f.endswith("pred.pcd")]
    # for i in range(len(predpcd_file)):
    #     pcd_f = predpcd_file[i]
    #     pcd = o3d.io.read_point_cloud(pcd_f)
    #     o3d.visualization.draw_geometries([pcd])


    # folder = "labelled"
    # epoch = 63
    # exp_folder = "pointnet2_real_data_0823_xyzrgb"
    # root_folder = f"eval_{folder}_{epoch}"
    # root_path = os.path.join(f"/home/minghan/workspace/plane_detection_NN/PointNet2_plane/log/plane_seg/{exp_folder}/", root_folder)
    # file_list = sorted(os.listdir(root_path))
    # gtpcd_file = [os.path.join(root_path, f) for f in file_list if f.endswith("gt.pcd")]
    # predpcd_file = [os.path.join(root_path, f) for f in file_list if f.endswith("pred.pcd")]
    # for i in range(len(predpcd_file)):
    #     pcd_f = gtpcd_file[i]
    #     pcd = o3d.io.read_point_cloud(pcd_f)
    #     o3d.visualization.draw_geometries([pcd])

    #     pcd_f = predpcd_file[i]
    #     pcd = o3d.io.read_point_cloud(pcd_f)
    #     o3d.visualization.draw_geometries([pcd])

    # log_dir = "pointnet2_real_data_0827_xyz"
    # prediction_dir = "eval_labelled_63"
    # root_path = f"/home/minghan/workspace/plane_detection_NN/PointNet2_plane/log/plane_seg/{log_dir}/{prediction_dir}"
    # file_list = sorted(os.listdir(root_path))
    # gtpcd_file = [os.path.join(root_path, f) for f in file_list if f.endswith("gt.pcd")]
    # predpcd_file = [os.path.join(root_path, f) for f in file_list if f.endswith("pred.pcd")]
    # for i in range(len(predpcd_file)):
    #     pcd_f = gtpcd_file[i]
    #     pcd = o3d.io.read_point_cloud(pcd_f)
    #     o3d.visualization.draw_geometries([pcd])

    #     pcd_f = predpcd_file[i]
    #     pcd = o3d.io.read_point_cloud(pcd_f)
    #     o3d.visualization.draw_geometries([pcd])

    # # # # TODO: visualize with different colours
    # cloud_path = "./data_scene/crop_data/cloud"
    # cloud_filename = sorted(os.listdir(cloud_path))
    # cloud_files = [os.path.join(cloud_path, filename) for filename in cloud_filename]

    # label_path = "./data_scene/crop_data/label"
    # label_filename = sorted(os.listdir(label_path))
    # label_files = [os.path.join(label_path, filename) for filename in label_filename]
    # assert len(cloud_files) == len(label_files)

    # for i in range(len(cloud_files)):
    #     # load point cloud
    #     pcd = o3d.io.read_point_cloud(cloud_files[i])
    #     points_num = np.asarray(pcd.points).shape[0]

    #     # load labels
    #     labels = np.load(label_files[i])

    #     # paint the point cloud
    #     plane_colors = np.array([[51/255.0, 160/255.0, 44/255.0]])
    #     non_plane_colors = np.array([[166/255.0, 206/255.0, 227/255.0]])
    #     gt_colors = np.repeat(non_plane_colors, points_num, axis=0)
    #     gt_colors[labels==1] = plane_colors
    #     pcd.colors = o3d.Vector3dVector(gt_colors)
    #     o3d.visualization.draw_geometries([pcd])
    # #     break

    # # TODO: Compare the results
    # rootpath = "./data_scene/results"
    # label_path = os.path.join(rootpath, "gt")
    # label_filename = sorted(os.listdir(label_path))
    # label_files = [os.path.join(label_path, filename) for filename in label_filename]

    # pointnet2_path = os.path.join(rootpath, "pointnet2")
    # pointnet2_filename = sorted(os.listdir(pointnet2_path))
    # pointnet2_files = [os.path.join(pointnet2_path, filename) for filename in pointnet2_filename]

    # ori_path = os.path.join(rootpath, "ori")
    # ori_filename = sorted(os.listdir(ori_path))
    # ori_files = [os.path.join(ori_path, filename) for filename in ori_filename]

    # assert len(label_files) == len(pointnet2_files) == len(ori_files)
    
    # total_num = 0
    # pointnet2_correct = 0
    # ori_correct = 0

    # num_class = [0, 0]
    # pointnet2_correct_class = [0, 0]
    # pointnet2_deno_class = [0, 0]
    # ori_correct_class = [0, 0]
    # ori_deno_class = [0, 0]
    # for i in range(len(label_files)):
    #     labels = np.load(label_files[i])
    #     total_num += labels.shape[0]
    #     num_class[0] += np.sum((labels == 0))
    #     num_class[1] += np.sum((labels == 1))

    #     pointnet2_pred = np.load(pointnet2_files[i])
    #     pointnet2_correct += (pointnet2_pred == labels).sum()
    #     pointnet2_correct_class[0] += np.sum((pointnet2_pred == 0) & (labels == 0))
    #     pointnet2_correct_class[1] += np.sum((pointnet2_pred == 1) & (labels == 1))
    #     pointnet2_deno_class[0] += np.sum((pointnet2_pred == 0) | (labels == 0))
    #     pointnet2_deno_class[1] += np.sum((pointnet2_pred == 1) | (labels == 1))

    #     with open(ori_files[i]) as f:
    #         lines = f.readlines()
    #     l_labels = [int(l[0]) if int(l[0]) == 0 else 1 for l in lines]
    #     ori_pred = np.asarray(l_labels)
    #     ori_correct += (ori_pred == labels).sum()
    #     ori_correct_class[0] += np.sum((ori_pred == 0) & (labels == 0))
    #     ori_correct_class[1] += np.sum((ori_pred == 1) & (labels == 1))
    #     ori_deno_class[0] += np.sum((ori_pred == 0) | (labels == 0))
    #     ori_deno_class[1] += np.sum((ori_pred == 1) | (labels == 1))

    # pointnet2_acc = (pointnet2_correct / total_num) * 100
    # ori_acc = (ori_correct / total_num) * 100
    # print('\n')
    # print(f'pointnet2 accuracy: {round(pointnet2_acc, 1)}%')
    # print(f'ori accuracy: {(round(ori_acc, 1))}%')

    # print('\n')
    # print(f'pointnet2 non-plane accuracy: {round((pointnet2_correct_class[0] / num_class[0]) * 100, 1)}%')
    # print(f'ori non-plane accuracy: {(round((ori_correct_class[0] / num_class[0]) * 100, 1))}%')

    # print('\n')
    # print(f'pointnet2 plane accuracy: {round((pointnet2_correct_class[1] / num_class[1]) * 100, 1)}%')
    # print(f'ori plane accuracy: {(round((ori_correct_class[1] / num_class[1]) * 100, 1))}%')

    # print('\n')
    # print(f'pointnet2 non-plane IOU: {round((pointnet2_correct_class[0] / pointnet2_deno_class[0]) * 100, 1)}%')
    # print(f'ori non-plane IOU: {(round((ori_correct_class[0] / ori_deno_class[0]) * 100, 1))}%')

    # print('\n')
    # print(f'pointnet2 plane IOU: {round((pointnet2_correct_class[1] / pointnet2_deno_class[1]) * 100, 1)}%')
    # print(f'ori plane IOU: {(round((ori_correct_class[1] / ori_deno_class[1]) * 100, 1))}%')

    # # pointnet2 classification
    # print((3.377694606781006 + 2.7212154865264893 + 2.6526401042938232 + 2.179804801940918 + 2.0946335792541504 + 2.7523531913757324) / 6)

    # # ori classification
    # print((1.345 + 0.968 + 1.67 + 1.543 + 0.82 + 1.382) / 6)

    # # ori 
    # print((2.484 + 1.885 + 3.12 + 2.62 + 1.742 + 2.046) / 6) 

    # # TODO: compute the percentage of planes and non-planes (gt)
    # rootpath = "./data_scene/crop_data"
    # label_path = os.path.join(rootpath, "label")
    # label_filename = sorted(os.listdir(label_path))
    # label_files = [os.path.join(label_path, filename) for filename in label_filename]

    # gt_plane_num = 0
    # gt_nonplane_num = 0
    # for i in range(len(label_files)):
    #     # load labels
    #     labels = np.load(label_files[i])

    #     gt_plane_num += labels[labels == 1].shape[0]
    #     gt_nonplane_num += labels[labels == 0].shape[0]
    # print(f"plane is {round(gt_plane_num * 100 / (gt_plane_num + gt_nonplane_num), 2)}% of gt label")
    # print(f"non-plane is {round(gt_nonplane_num * 100 / (gt_plane_num + gt_nonplane_num), 2)}% of gt label")

    # # TODO: build new scene
    # rootpath1 = "./data_synthetic/pcd_plane"
    # rootpath2 = "./data_synthetic/pcd_nonplane"
    # train_ratio1 = 0.9
    # train_ratio2 = 0.9
    # mod = "train"
    # # loading data with planes (dataset 1)
    # cloud_path1 = os.path.join(rootpath1, "pcd_noise")
    # cloud_filename1 = sorted(os.listdir(cloud_path1))
    # cloud_files1 = [os.path.join(cloud_path1, filename) for filename in cloud_filename1]
    
    # label_path1 = os.path.join(rootpath1, "label")
    # label_filename1 = sorted(os.listdir(label_path1))
    # label_files1 = [os.path.join(label_path1, filename) for filename in label_filename1]
    # assert len(cloud_files1) == len(label_files1)

    # # split data with planes (dataset 1)
    # model_num1 = len(cloud_files1)
    # train_size1 = int(model_num1 * train_ratio1)
    # indices1 = list(range(model_num1))
    # random.seed(4)
    # random.shuffle(indices1)
    # if mod == "train":
    #     split_indices1 = indices1[:train_size1][:10]
    # elif mod == "test":
    #     split_indices1 = indices1[train_size1:][:10]
    # else:
    #     raise Exception("mod should be train or test")
    # print(f"loading {len(split_indices1)} models ...")

    # plane_points_list = []
    # plane_label_list = []
    # for i in tqdm(split_indices1, total=len(split_indices1)):
    #     # load point cloud
    #     pcd = o3d.io.read_point_cloud(cloud_files1[i])
    #     points = np.asarray(pcd.points)
    #     plane_points_list.append(points)

    #     # load labels
    #     labels = np.load(label_files1[i]).astype(np.float64)
    #     plane_label_list.append(labels)

    # # loading data with non-planes (dataset 2)
    # cloud_path2 = os.path.join(rootpath2, "pcd_noise")
    # cloud_filename2 = sorted(os.listdir(cloud_path2))
    # cloud_files2 = [os.path.join(cloud_path2, filename) for filename in cloud_filename2]
    
    # label_path2 = os.path.join(rootpath2, "label")
    # label_filename2 = sorted(os.listdir(label_path2))
    # label_files2 = [os.path.join(label_path2, filename) for filename in label_filename2]
    # assert len(cloud_files2) == len(label_files2)

    # # split data with non-planes (dataset 2)
    # model_num2 = len(cloud_files2)
    # train_size2 = int(model_num2 * train_ratio2)
    # indices2 = list(range(model_num2))
    # random.seed(4)
    # random.shuffle(indices2)
    # if mod == "train":
    #     split_indices2 = indices2[:train_size2]
    # elif mod == "test":
    #     split_indices2 = indices2[train_size2:]
    # else:
    #     raise Exception("mod should be train or test")
    # print(f"loading {len(split_indices2)} models ...")

    # for i in tqdm(split_indices2, total=len(split_indices2)):
    #     # load point cloud
    #     pcd = o3d.io.read_point_cloud(cloud_files2[i])
    #     points = np.asarray(pcd.points)
    #     points = points - np.mean(points, axis=0)
    #     coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

    #     # load labels
    #     labels = np.load(label_files2[i]).astype(np.float64)

    #     # choose planes
    #     plane_idx = np.random.choice(np.arange(len(plane_points_list)), 5, replace=False)
    #     plane_points_0 = plane_points_list[plane_idx[0]]
    #     plane_points_1 = plane_points_list[plane_idx[1]]
    #     plane_points_2 = plane_points_list[plane_idx[2]]
    #     plane_points_3 = plane_points_list[plane_idx[3]]
    #     plane_points_4 = plane_points_list[plane_idx[4]]

    #     plane_labels_0 = plane_label_list[plane_idx[0]]
    #     plane_labels_1 = plane_label_list[plane_idx[1]]
    #     plane_labels_2 = plane_label_list[plane_idx[2]]
    #     plane_labels_3 = plane_label_list[plane_idx[3]]
    #     plane_labels_4 = plane_label_list[plane_idx[4]]

    #     offset_x = (coord_max[0]*0.5 - coord_min[0]) * np.random.random_sample() + coord_min[0]
    #     offset_y = (coord_max[1]*0.5 - coord_min[1]) * np.random.random_sample() + coord_min[1]
    #     plane_points_0 = plane_points_0 - np.mean(plane_points_0, axis=0) + np.array([0, 0, coord_max[2]])
    #     plane_points_1 = plane_points_1 - np.mean(plane_points_1, axis=0) + np.array([0, offset_y, coord_max[2]])
    #     plane_points_2 = plane_points_2 - np.mean(plane_points_2, axis=0) + np.array([offset_x, 0, coord_max[2]])
    #     plane_points_3 = plane_points_3 - np.mean(plane_points_3, axis=0) + coord_max
    #     plane_points_4 = plane_points_4 - np.mean(plane_points_4, axis=0) + coord_min

    #     combined_points = np.concatenate((points, plane_points_0, plane_points_1, plane_points_2, plane_points_3, plane_points_4))
    #     combined_labels = np.concatenate((labels, plane_labels_0, plane_labels_1, plane_labels_2, plane_labels_3, plane_labels_4))
    #     points_num = combined_points.shape[0]
    #     assert combined_points.shape[0] == combined_labels.shape[0]
    #     print((combined_points.shape[0] - points.shape[0]) / combined_points.shape[0])

    #     # colors
    #     plane_colors = np.array([[0.1, 0.1, 0.3]])
    #     non_plane_colors = np.array([[0.8, 0.2, 0.3]])
    #     combined_colors = np.repeat(non_plane_colors, points_num, axis=0)
    #     combined_colors[combined_labels==1] = plane_colors
    #     # create point cloud
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(combined_points)
    #     pcd.colors = o3d.Vector3dVector(combined_colors)
    #     o3d.visualization.draw_geometries([pcd])
    #     # break
