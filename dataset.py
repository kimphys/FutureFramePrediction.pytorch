import torch
from torch.utils.data import Dataset

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

import os
import glob

class SequenceDataset(Dataset):
    def __init__(self, channels, size, videos_dir, time_steps):

        self.videos = []
        self.seqs_idx = []

        for f in sorted(os.listdir(videos_dir)):
            frames = glob.glob(os.path.join(videos_dir, f, '*.tif'), recursive=True)
            frames.sort()
            self.videos.append(frames)

            selected_idx = np.random.choice(len(frames) - time_steps, size=5)

            self.seqs_idx.append(selected_idx)

        self.time_steps = time_steps
        self.size = size

        self.channels = channels

    def __len__(self):                    
        return len(self.videos)

    def __getitem__(self, index):

        video = self.videos[index]

        selected_idx = self.seqs_idx[index]

        clips = []
        for idx in selected_idx:
            frames = video[idx:idx + self.time_steps]

            if self.channels == 1:
                frames = [cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32) for frame in frames]
            else: 
                frames = [cv2.imread(frame, cv2.IMREAD_COLOR).astype(np.float32) for frame in frames]
                    
            frames = [simple_transform(frame, self.size, self.channels) for frame in frames]

            frames = torch.stack(frames)
            frames = torch.reshape(frames, (-1, self.size, self.size))
            clips.append(frames)

        return clips


class TestDataset(Dataset):
    def __init__(self, channels, size, videos_dir, time_steps):

        self.videos = glob.glob(os.path.join(videos_dir, '*.tif'), recursive=True)
        self.videos.sort()

        self.time_steps = time_steps
        self.size = size

        self.channels = channels

    def __len__(self):                    
        return len(self.videos) - self.time_steps

    def __getitem__(self, index):  

        if self.channels == 1:
            frames = [cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32) for frame in self.videos[index:index + self.time_steps]]
        else: 
            frames = [cv2.imread(frame, cv2.IMREAD_COLOR).astype(np.float32) for frame in self.videos[index:index + self.time_steps]]

        o_seqs = [base_transform(img, self.size) for img in frames]
        seqs = [simple_transform(img, self.size, self.channels) for img in frames]
        
        seqs = torch.stack(seqs)
        seqs = torch.reshape(seqs, (-1, self.size, self.size))

        return seqs, o_seqs

def simple_transform(img, size, channels):

    if channels == 1:
        mean = 0.5
        std = 0.5
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = A.Compose([
                            A.Resize(height=size, 
                                    width=size, 
                                    always_apply=True, 
                                    p=1.0),
                            A.Normalize(mean=mean,
                                        std=std,
                                        max_pixel_value=255.0,
                                        p=1.0),
                            ToTensorV2(p=1.0)
                        ], p=1.0)

    img = transform(image=img)['image']

    return img

def base_transform(img, size):
    transform = A.Compose([
                            A.Resize(height=size, 
                                    width=size, 
                                    always_apply=True, 
                                    p=1.0)
                        ], p=1.0)

    img = transform(image=img)['image']

    return img

    