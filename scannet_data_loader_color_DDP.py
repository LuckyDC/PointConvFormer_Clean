from turtle import color
import open3d as o3d
import sys
import os

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import pickle
from sklearn.neighbors import KDTree
import random
from easydict import EasyDict as edict
import yaml
from torch.utils.data import Dataset
import transforms as t
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
from util.voxelize import voxelize

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    #method = "voxelcenters" # "barycenters" "voxelcenters"
    method = "barycenters"

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose, method=method)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose, method=method)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose, method=method)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose, method=method)

def compute_weight(train_data, num_class = 20):
    weights = np.array([0.0 for i in range(num_class)])

    num_rooms = len(train_data)
    for i in range(num_rooms):
        _, _, labels,_ = train_data[i]
        #rm invalid labels
        labels = labels[labels >= 0]
        for j in range(num_class):
            weights[j] += np.sum(labels == j)

    ratio = weights / float(sum(weights))
    # ce_label_weight = 1 / (np.power(ratio, 1/3))
    ce_label_weight = 1 / (np.power(ratio, 1/2))
    return list(ce_label_weight)

def compute_knn(ref_points, query_points, K, dialated_rate = 1):
    num_ref_points = ref_points.shape[0]

    if num_ref_points < K or num_ref_points < dialated_rate * K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(num_ref_points, (num_query_points, K)).astype(np.int32)

        return inds

    kdt = KDTree(ref_points)
    neighbors_idx = kdt.query(query_points, k = K * dialated_rate, return_distance=False)
    neighbors_idx = neighbors_idx[:, ::dialated_rate]

    return neighbors_idx

def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


class ScanNetDataset(Dataset):
    def __init__(self, cfg, set="training", rotate_aug=False, flip_aug=False, scale_aug=False, transform_aug=False, color_aug=False, crop=False, shuffle_index=False):

        self.data = []
        self.cfg = cfg
        self.set = set

        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform = transform_aug
        # self.trans_std = [0.1, 0.1, 0.1]
        self.trans_std = [0.02, 0.02, 0.02]
        self.color_aug = color_aug
        self.crop = crop
        self.shuffle_index = shuffle_index

        if self.color_aug:
            '''
            color_transform = [t.ChromaticAutoContrast(),
                               t.ChromaticTranslation(0.1),
                               t.ChromaticJitter(0.05)]
            '''
            
            color_transform = [t.RandomDropColor()]
            self.color_transform = t.Compose(color_transform)
        
        if set == "training":
            data_files = glob.glob(self.cfg.train_data_path)
        elif set == "validation":
            data_files = glob.glob(self.cfg.val_data_path)
        elif set == "trainval":
            data_files_train = glob.glob(self.cfg.train_data_path)
            data_files_val = glob.glob(self.cfg.val_data_path)
            data_files = data_files_train + data_files_val
        else:
            raise ValueError('working on test set: ', self.set)


        for x in torch.utils.data.DataLoader(
                data_files,
                collate_fn=lambda x: torch.load(x[0]), num_workers=cfg.NUM_WORKERS):
            self.data.append(x)

        print('%s examples: %d'%(self.set, len(self.data)))
 
        if self.cfg.USE_WEIGHT:
            weights = compute_weight(self.data)
        else:
            weights = [1.0] * 20
        print("label weights", weights)
        self.cfg.weights = weights

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.data)

    def _augment_data(self, coord, color, norm, label):
        #########################
        # random data augmentation 
        #########################

        # random augmentation by rotation
        if self.rotate_aug:
            # rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            rotate_rad = np.deg2rad(torch.rand(1).numpy()[0] * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            coord[:, :2] = np.dot(coord[:, :2], j)
            norm[:, :2] = np.dot(norm[:, :2], j)
            # print(rotate_rad)

        # random augmentation by flip x, y or x+y
        if self.flip_aug:
            flip_type = torch.randint(4, (1,)).numpy()[0] # np.random.choice(4, 1)
            # print(flip_type)
            if flip_type == 1:
                coord[:, 0] = -coord[:, 0]
                norm[:, 0] = -norm[:, 0]
            elif flip_type == 2:
                coord[:, 1] = -coord[:, 1]
                norm[:, 1] = -norm[:, 1]
            elif flip_type == 3:
                coord[:, :2] = -coord[:, :2]
                norm[:, :2] = -norm[:, :2]

        if self.scale_aug:
            noise_scale = torch.rand(1).numpy()[0] * 0.4 + 0.8 # np.random.uniform(0.8, 1.2)
            # print(noise_scale)
            coord[:, 0] = noise_scale * coord[:, 0]
            coord[:, 1] = noise_scale * coord[:, 1]

        if self.transform:
            # noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
            #                             np.random.normal(0, self.trans_std[1], 1),
            #                             np.random.normal(0, self.trans_std[2], 1)]).T
            # noise_translate = np.array([torch.randn(1).numpy()[0] * self.trans_std[0],
            #                             torch.randn(1).numpy()[0] * self.trans_std[1],
            #                             torch.randn(1).numpy()[0] * self.trans_std[2]]).T
            num_points = coord.shape[0]
            noise_translate = torch.randn(num_points, 3).numpy()
            noise_translate[:, 0] *= self.trans_std[0]
            noise_translate[:, 1] *= self.trans_std[1]
            noise_translate[:, 2] *= self.trans_std[2]
            # print("before range: ", coord.min(0), coord.max(0))
            coord[:, 0:3] += noise_translate
            # print("after range: ", coord.min(0), coord.max(0))
            # print(noise_translate)

        if self.color_aug:
            # color = (color + 1) * 127.5
            # color *= 255.
            coord, color, label, norm = self.color_transform(coord, color, label, norm)
            # color = color / 127.5 - 1
            # color /= 255.

        # crop half of the scene
        if self.crop:
            points = coord - coord.mean(0)
            if torch.rand(1).numpy()[0] < 0.5:
                inds = np.all([points[:, 0] >= 0.0], axis = 0)
            else:
                inds = np.all([points[:, 0] < 0.0], axis = 0)
            
            coord, color, norm, label = (
                coord[~inds],
                color[~inds],
                norm[~inds],
                label[~inds]
            )

        return coord, color, norm, label

    def __getitem__(self, indx):

        coord, features, label, scene_name = self.data[indx]

        color, norm = features[:, :3], features[:, 3:]

        # move z to 0+
        z_min = coord[:, 2].min()
        coord[:, 2] -= z_min 

        coord, color, norm, label = self._augment_data(coord, color, norm, label)

        # voxelize coords
        if self.set in ["training", "validation"]:
            coord_min = np.min(coord, 0)
            coord -= coord_min 
            uniq_idx = voxelize(coord, self.cfg.grid_size[0], mode=self.set)
            coord, color, norm, label = coord[uniq_idx], color[uniq_idx], norm[uniq_idx], label[uniq_idx]

        # subsample points
        if (self.set == "training" or self.set == "trainval") and label.shape[0] > self.cfg.MAX_POINTS_NUM:
            init_idx = torch.randint(label.shape[0], (1,)).numpy()[0]
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:self.cfg.MAX_POINTS_NUM]
            coord, color, norm, label = coord[crop_idx], color[crop_idx], norm[crop_idx], label[crop_idx]

        # shuffle_index
        if self.shuffle_index:
            shuf_idx = torch.randperm(coord.shape[0]).numpy()
            coord, color, norm, label = coord[shuf_idx], color[shuf_idx], norm[shuf_idx], label[shuf_idx]

        # input normalize
        coord_min = np.min(coord, 0)
        coord -= coord_min

        # print("number of points: ", coord.shape[0])
        point_list, color_list, label_list, norm_list = [], [], [], []
        nei_forward_list, nei_propagate_list, nei_self_list = [], [], []

        for j, grid_s in enumerate(self.cfg.grid_size):
            if j == 0:
                # sub_point, sub_norm_color, sub_labels = \
                #     grid_subsampling(points=coord.astype(np.float32), features=np.concatenate((norm, color), axis=1).astype(np.float32), \
                #                      labels=label.astype(np.int32), sampleDl=grid_s)
                sub_point, sub_color, sub_norm, sub_labels = coord.astype(np.float32), color.astype(np.float32), norm.astype(np.float32), \
                    label.astype(np.int32)
                
                point_list.append(sub_point)
                color_list.append(sub_color)
                norm_list.append(sub_norm)
                label_list.append(sub_labels)

                # compute edges
                nself = compute_knn(sub_point, sub_point, self.cfg.K_self[j])
                nei_self_list.append(nself)

            else:
                sub_point, sub_norm = \
                    grid_subsampling(points=point_list[-1], features=norm_list[-1], sampleDl=grid_s)

                if sub_point.shape[0] <= self.cfg.K_self[j]:
                    sub_point, sub_norm = point_list[-1], norm_list[-1]

                # compute edges
                nforward = compute_knn(point_list[-1], sub_point, self.cfg.K_forward[j])
                npropagate = compute_knn(sub_point, point_list[-1], self.cfg.K_propagate[j])
                nself = compute_knn(sub_point, sub_point, self.cfg.K_self[j])

                point_list.append(sub_point)
                norm_list.append(sub_norm)
                nei_forward_list.append(nforward)
                nei_propagate_list.append(npropagate)
                nei_self_list.append(nself)

        return color_list, point_list, nei_forward_list, nei_propagate_list, nei_self_list, label_list, norm_list

def tensorlizeList(nplist, is_index = False):
    ret_list = []
    for i in range(len(nplist)):
        if is_index:
            if nplist[i] is None:
                ret_list.append(None)
            else:
                ret_list.append(torch.from_numpy(nplist[i]).long().unsqueeze(0))
        else:
            ret_list.append(torch.from_numpy(nplist[i]).float().unsqueeze(0))

    return ret_list

def tensorlize(features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms):
    pointclouds = tensorlizeList(pointclouds)
    norms = tensorlizeList(norms)
    edges_self = tensorlizeList(edges_self, True)
    edges_forward = tensorlizeList(edges_forward, True)
    edges_propagate = tensorlizeList(edges_propagate, True)

    target = torch.from_numpy(target).long().unsqueeze(0)
    features = torch.from_numpy(features).float().unsqueeze(0)

    return features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms

def listToBatch(features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms):
    # import ipdb; ipdb.set_trace()
    num_sample = len(pointclouds)

    #process sample 0
    featureBatch = features[0][0]
    pointcloudsBatch = pointclouds[0]
    pointcloudsNormsBatch = norms[0]
    targetBatch = target[0][0]

    edgesSelfBatch = edges_self[0]
    edgesForwardBatch = edges_forward[0]
    edgesPropagateBatch = edges_propagate[0]

    points_stored = [val.shape[0] for val in pointcloudsBatch]

    for i in range(1, num_sample):
        targetBatch = np.concatenate([targetBatch, target[i][0]], 0)
        featureBatch = np.concatenate([featureBatch, features[i][0]], 0)

        for j in range(len(edges_forward[i])):
            tempMask = edges_forward[i][j] == -1
            edges_forwardAdd = edges_forward[i][j] + points_stored[j]
            edges_forwardAdd[tempMask] = -1
            edgesForwardBatch[j] = np.concatenate([edgesForwardBatch[j], \
                                   edges_forwardAdd], 0)

            tempMask2 = edges_propagate[i][j] == -1
            edges_propagateAdd = edges_propagate[i][j] + points_stored[j + 1]
            edges_propagateAdd[tempMask2] = -1
            edgesPropagateBatch[j] = np.concatenate([edgesPropagateBatch[j], \
                                   edges_propagateAdd], 0)

        for j in range(len(pointclouds[i])):
            tempMask3 = edges_self[i][j] == -1
            edges_selfAdd = edges_self[i][j] + points_stored[j]
            edges_selfAdd[tempMask3] = -1
            edgesSelfBatch[j] = np.concatenate([edgesSelfBatch[j], \
                                    edges_selfAdd], 0)

            pointcloudsBatch[j] = np.concatenate([pointcloudsBatch[j], pointclouds[i][j]], 0)
            pointcloudsNormsBatch[j] = np.concatenate([pointcloudsNormsBatch[j], norms[i][j]], 0)

            points_stored[j] += pointclouds[i][j].shape[0]

    return featureBatch, pointcloudsBatch, edgesSelfBatch, edgesForwardBatch, edgesPropagateBatch, \
           targetBatch, pointcloudsNormsBatch

def prepare(features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms):

    features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out = [], [], [], [], [], [], []

    features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out = \
        listToBatch(features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms)

    features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out = \
        tensorlize(features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out)

    return features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out

def collect_fn(data_list):
    # import ipdb; ipdb.set_trace()
    features = []
    pointclouds = []
    target = []
    norms = []
    edges_forward = []
    edges_propagate = []
    edges_self = []
    for idx, data in enumerate(data_list):

        feature_list, point_list, nei_forward_list, nei_propagate_list, nei_self_list, label_list, normal_list = data
        features.append(feature_list)
        pointclouds.append(point_list)
        target.append(label_list)
        norms.append(normal_list)

        edges_forward.append(nei_forward_list)
        edges_propagate.append(nei_propagate_list)
        edges_self.append(nei_self_list)

    features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = \
            prepare(features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms)

    return features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms

def getdataLoadersDDP(cfg):
    # Initialize datasets
    training_dataset = ScanNetDataset(cfg, set="training", 
                                      rotate_aug=True,
                                      flip_aug=False,
                                      scale_aug=True,
                                      transform_aug=False,
                                      color_aug=True,
                                      crop=False, 
                                      shuffle_index=True)
    validation_dataset = ScanNetDataset(cfg, set="validation")

    training_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)

    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)

    train_data_loader = torch.utils.data.DataLoader(training_dataset,
                                                    batch_size=cfg.BATCH_SIZE, 
                                                    collate_fn=collect_fn, 
                                                    num_workers=cfg.NUM_WORKERS, 
                                                    pin_memory=True,
                                                    sampler=training_sampler,
                                                    drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                  batch_size=cfg.BATCH_SIZE, 
                                                  collate_fn=collect_fn, 
                                                  num_workers=cfg.NUM_WORKERS, 
                                                  sampler=validation_sampler,
                                                  pin_memory=True)

    return train_data_loader, val_data_loader



