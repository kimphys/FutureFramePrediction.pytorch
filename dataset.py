import torch
from torch.utils.data import Dataset

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

import os
import glob

class SequenceDataset(Dataset):
    def __init__(self, channels, size, videos_dir, time_steps, test=False):
        self.test = test

        self.videos = []

        for f in sorted(os.listdir(videos_dir)):
            frames = glob.glob(os.path.join(videos_dir, f, '*.tif'), recursive=True)
            frames.sort()
            self.videos.append(frames)

        self.time_steps = time_steps
        self.size = size

        self.channels = channels

    def __len__(self):                    
        return len(self.videos)

    def __getitem__(self, index):

        video = self.videos[index]

        selected_idx = np.random.choice(len(video) - self.time_steps, size=10)

        clips = []
        for idx in selected_idx:
            frames = video[idx:idx + self.time_steps]

            if self.channels == 1:
                frames = [cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32) for frame in frames]
            else: 
                frames = [cv2.imread(frame, cv2.IMREAD_COLOR).astype(np.float32) for frame in frames]
                    
            frames = [self.simple_transform(frame, self.size) for frame in frames]

            frames = torch.stack(frames)
            frames = torch.reshape(frames, (-1, self.size, self.size))
            clips.append(frames)

        return clips

    def simple_transform(self, img, size):

        if self.channels == 1:
            mean = 0.5
            std = 0.5
        else:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]

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

    