Experiment description: 1164 planes and 241 non-planes for training, 4 ori example data (referred as "labelled") and 5 lidar data (referred as "unlabelled") collected in the lab for whole scene evaluation

pointnet2_sem_seg.txt: training details in each epoch

eval.txt: evaluation details

plots folder:
Accuracy.png: accuracy during trainig and testing
Mean Loss.png: mean loss during training and testing
IOU: iou during training

pcd_visual folder:
predicted pcd
"labelled" means evaluation with 4 ori example data, using oxf package to label the point cloud. Blue for plane, pink for non-plane.
"unlabelled" means evaluation with 5 lidar data collected in the lab. Blue for plane, pink for non-plane.
"epochX" means the number of training epochs X

