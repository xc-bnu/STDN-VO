import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from datasets.utils import rotation_to_euler
import torch
import matplotlib.pyplot as plt

class KITTI(torch.utils.data.Dataset):

    def __init__(self,
                 data_path=r"data/sequences_jpg",
                 gt_path=r"data/poses",
                 camera_id="2",
                 sequences=["00", "01", "02", "03", "04", "05", "06", "07", "08"],
                 window_size=3,
                 overlap=1,
                 read_poses=True,
                 transform=None,
                 train_mode=False,
                 ):


        self.data_path = data_path
        self.gt_path = gt_path
        self.camera_id = camera_id
        self.frame_id = 0
        self.read_poses = read_poses
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform
        self.train_mode = train_mode

        # KITTI normalization
        self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        # define sequence for training, test and val
        self.sequences = sequences

        # read frames list and ground truths
        frames, seqs = self.read_frames()
        seq_len = {
            '11': 921,
            '12': 1061,
            '13': 3281,
            '14': 631,
            '15': 1901,
            '16': 1731,
            '17': 491,
            '18': 1801,
            '19': 4981,
            '20': 831,
            '21': 2721,
            'test': 2,
        }
        if self.train_mode:
            gt = self.read_gt()
        else:
            gt = [None] * seq_len[self.sequences[0]] 

        # create dataframe with frames and ground truths
        data = pd.DataFrame({"gt": gt}) 
        data = data["gt"].apply(pd.Series) 
        data["frames"] = frames
        data["sequence"] = seqs 
        self.data = data
        self.windowed_data = self.create_windowed_dataframe(data)

    def __len__(self):
        return len(self.windowed_data["w_idx"].unique())

    def __getitem__(self, idx):

        # get data of corresponding window index
        data = self.windowed_data.loc[self.windowed_data["w_idx"] == idx, :] 

        # Read frames as grayscale
        frames = data["frames"].values
        imgs = []
        for fname in frames:
            img = Image.open(fname).convert('RGB')
            img = self.transform(img) 
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.asarray(imgs)
        imgs = imgs.transpose(1, 0, 2, 3)

        if self.train_mode:
            gt_poses = data.loc[:, [i for i in range(12)]].values
            y = []
            for gt_idx, gt in enumerate(gt_poses):

                pose = np.vstack([np.reshape(gt, (3, 4)), [[0., 0., 0., 1.]]])

                # compute relative pose from frame1 to frame2
                if gt_idx > 0:
                    pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose)
                    R = pose_wrt_prev[:3, :3]
                    t = pose_wrt_prev[:3, 3]

                    # Euler parameterization (rotations as Euler angles)
                    angles = rotation_to_euler(R, seq='zyx')

                    # normalization
                    angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
                    t = (np.asarray(t) - self.mean_t) / self.std_t

                    # concatenate angles and translation
                    y.append(list(angles) + list(t))

                pose_prev = pose

            y = np.asarray(y)
            y = y.flatten()

            return imgs, y
        else:
            return imgs, np.array([])

    def read_frames(self):

        frames = [] 
        seqs = [] 

        for sequence in self.sequences:
            frames_dir = os.path.join(self.data_path, sequence, "image_{}".format(self.camera_id), "*.jpg")
            frames_seq = sorted(glob.glob(frames_dir))
            frames = frames + frames_seq
            seqs = seqs + [sequence] * len(frames_seq)
        return frames, seqs

    def read_gt(self):

        if self.read_gt:
            gt = []
            for sequence in self.sequences:
                with open(os.path.join(self.gt_path, sequence + ".txt")) as f:
                    lines = f.readlines()

                # convert poses to float
                for line_idx, line in enumerate(lines):
                    line = line.strip().split()
                    line = [float(x) for x in line]
                    gt.append(line)

        else:
            gt = None

        return gt

    def create_windowed_dataframe(self, df):
        window_size = self.window_size 
        overlap = self.overlap 
        windowed_df = pd.DataFrame()
        w_idx = 0

        for sequence in df["sequence"].unique():
            seq_df = df.loc[df["sequence"] == sequence, :].reset_index(drop=True)
            row_idx = 0
            while row_idx + window_size <= len(seq_df):
                rows = seq_df.iloc[row_idx:(row_idx + window_size)].copy() 
                rows["w_idx"] = len(rows) * [w_idx]  
                row_idx = row_idx + window_size - overlap
                w_idx = w_idx + 1
                windowed_df = pd.concat([windowed_df, rows], ignore_index=True)
        windowed_df.reset_index(drop=True)
        return windowed_df
