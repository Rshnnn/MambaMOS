"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import re

import random

from .builder import DATASETS
from .defaults import DefaultMultiScansDataset

def points_transform(points, from_pose, to_pose):
  transformation = np.linalg.inv(to_pose).dot(from_pose)
  points = np.hstack((points, np.ones((points.shape[0], 1)))).T

  return transformation.dot(points).T[:, :3]

def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
        Args:
            pose_path: (Complete) filename for the pose file
        Returns:
            A numpy array of size nx4x4 with n poses as 4x4 transformation
            matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)

def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)

def absoluteFilePaths(directory, selected_file_list=None):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if selected_file_list is None or f.split('.')[0] in selected_file_list:
                yield os.path.abspath(os.path.join(dirpath, f))

@DATASETS.register_module()
class SemanticKITTIMultiScansDataset(DefaultMultiScansDataset):
    def __init__(
        self,
        split="train",
        data_root="data",
        data_width = 460,
        data_height = 352,
        gather_num=6,
        scan_modulation=False,
        transform=None,
        test_mode=False,
        test_cfg=None,
        windows_stride=None,
        loop=1,
        ignore_index=-1,
    ):
        self.gather_num = gather_num
        self.data_width = data_width
        self.data_height = data_height
        self.scan_modulation = scan_modulation
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            gather_num=gather_num,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

        self.windows_stride = windows_stride

    def get_pose_data(self, pose_file, calib_file):
        # load poses
        poses = np.array(load_poses(pose_file))
        inv_frame0 = np.linalg.inv(poses[0])

        # load calibrations
        T_cam_velo = load_calib(calib_file)
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)

        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
            new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))

        return np.array(new_poses)

    def filter_dataset(self, seq_list):
        with open(os.path.join(os.path.dirname(__file__), "./train_split_dynamic_pointnumber.txt")) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]

        pending_dict = {}
        # num_dict = {}
        for line in lines:
            if line != '':
                # seq, fid, moving_points_num = line.split()
                seq, fid, moving_points_num = line.split()
                if int(seq) in seq_list:
                    if seq in pending_dict.keys():
                        if fid in pending_dict[seq]:
                            raise ValueError(f"!!!! Duplicate {fid} in seq {seq} in .txt file")
                        pending_dict[seq].append(fid)
                    else:
                        pending_dict[seq] = [fid]
        return pending_dict

    def get_data_list(self):
        split2seq = dict(
            # train=[1, 10, 15, 17, 25, 40, 75, 80, 100, 125, 175, 200],
            # val=[50],
            # test=[5, 20, 30, 60, 150],
            train=[1, 3],
            val=[2],
            test=[4],

        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        self.poses = {}
        pending_dict = self.filter_dataset(seq_list)
        for seq in seq_list:
            seq = f"rain_{seq}"
            # seq = str(seq)+"mm"
            # seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_folder = os.path.join(self.data_root,  "merge_data")
            if self.split == "train":
                data_list += absoluteFilePaths(os.path.join(seq_folder, seq))
                print(f"Processing all frames in Seq {seq} without filtering.")
            else:
                data_list += absoluteFilePaths(os.path.join(seq_folder, seq))
        return data_list

    def get_multi_data(self, idx):
        cur_data_path = self.data_list[idx % len(self.data_list)]

        multi_scan_path, gather_coord, gather_strength, gather_segment = [], [], [], []
        seq, _, file_name = cur_data_path.split('/')[-3:]

        cur_scan_index = int(file_name.split('.')[0])

        tn = []
        modulation = 1
        if self.scan_modulation:
            scan_modulation_prob = random.random()
            if 0.5 < scan_modulation_prob <= 0.75:
                modulation = 2
            elif scan_modulation_prob > 0.75:
                modulation = 3
            if self.split != "train":
                modulation = 3

        for i in range(self.gather_num):
            last_scan_index = cur_scan_index - modulation * i
            last_scan_index = max(0, last_scan_index)
            scan_path = cur_data_path.replace(cur_data_path.split("/")[-1], f"{str(last_scan_index).zfill(10)}.npz")
            if not os.path.exists(scan_path):
                print(f"File {scan_path} does not exist, skipping...")
                continue

            # 读取 `.npz` 数据
            with np.load(scan_path) as data:
                x = data['x'] 
                y = data['y'] 
                t = data['t']
                p = data['p']

                # 构建 coord 和 strength
                coord = np.stack((x, y, t), axis=-1)
                strength = p.reshape(-1, 1)
            # print(f"coord: {coord.shape}, strength: {strength.shape}")
            # 读取 segment 数据
            segment_path = scan_path.replace("merge_data", "raw_data")
            segment_path = re.sub(r"/rain_\d+/", "/", segment_path)
            # segment_path = re.sub(r"/\d+mm/", "/", segment_path)
            # print(f"segment_path: {segment_path}")
            if os.path.exists(segment_path):
                with np.load(segment_path) as segment_data:
                    segment_x = segment_data['x'] 
                    segment_y = segment_data['y'] 
                    segment_t = segment_data['t']
                    segment_p = segment_data['p']

                    # 归一化 segment_t
                    segment_t_normalized = (segment_t - segment_t.min()) / (segment_t.max() - segment_t.min())

                    # 构建 segment
                    segment = np.stack((segment_x, segment_y, segment_t_normalized, segment_p), axis=-1)
                    # print(f"segment shape: {segment.shape}")

            else:
                print(f"segment_path: {segment_path}")
                segment = np.zeros(coord.shape).astype(np.float32)
                # print(f")))))))Processing segment: {segment.shape}")
            # 归一化时间 t
            t_normalized = (t - t.min()) / (t.max() - t.min())  # 归一化 t 到 [0, 1] 范围

            # 将归一化的时间 t 添加到 coord 中
            coord = np.column_stack((x, y, t_normalized))  # 组合 (x, y, t_normalized)

            # 添加时间窗口索引
            # print(f"Processing scan file: {scan_path}")
            tn.append(np.ones(coord.shape[0]) * i)  # i 为窗口索引

            # 聚合当前文件的数据
            gather_coord.append(coord)
            gather_strength.append(strength)
            gather_segment.append(segment)
            multi_scan_path.append(scan_path)

        # print(gather_coord.shape)
        # print(gather_coord)
        # print(gather_coord[0])
        # print(gather_coord[4])
        # print(gather_coord[5])
        # print(gather_segment.shape)
        data_dict = dict(coord=np.concatenate(gather_coord), strength=np.concatenate(gather_strength),
                        segment=np.concatenate(gather_segment), tn=np.expand_dims(np.concatenate(tn), axis=1))

        return data_dict

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]

        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        '''     
        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        '''

        segment_path = data_path.replace("velodyne", "segments").replace(".bin", ".segment")
        if os.path.exists(segment_path):
            with open(segment_path, "rb") as a:
                segment = np.fromfile(a, dtype=np.float32).reshape(-1, 4)
        else:
            segment = np.zeros(scan.shape).astype(np.float32)

        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"

        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            9: 1,
            10: 2,  # "car"
            11: 2,  # "bicycle"
            13: 2,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 2,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 2,  # "truck"
            20: 2,  # "other-vehicle"
            30: 2,  # "person"
            31: 2,  # "bicyclist"
            32: 2,  # "motorcyclist"
            40: 1,  # "road"
            44: 1,  # "parking"
            48: 1,  # "sidewalk"
            49: 1,  # "other-ground"
            50: 1,  # "building"
            51: 1,  # "fence"
            52: 1,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 1,  # "lane-marking" to "road" ---------------------------------mapped
            70: 1,  # "vegetation"
            71: 1,  # "trunk"
            72: 1,  # "terrain"
            80: 1,  # "pole"
            81: 1,  # "traffic-sign"
            99: 1,  # "other-object" to "unlabeled" ----------------------------mapped
            250: 2,
            251: 3,
            252: 3,  # "moving-car" to "car" ------------------------------------mapped
            253: 3,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 3,  # "moving-person" to "person" ------------------------------mapped
            255: 3,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 3,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 3,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 3,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }

        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            1: 9,
            2: 250,
            3: 251,
        }

        return learning_map_inv
