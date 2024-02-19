import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

import numpy as np

class KITTIImageEnhanceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        
        self.input_images = sorted(os.listdir(os.path.join(root_dir, 'Input')))
        self.gt_images = sorted(os.listdir(os.path.join(root_dir, 'GT')))

        self.img_size=self.read_img(os.path.join(self.root_dir, 'Input', self.input_images[0])).shape[:2]
        self.scale_size=self.img_size

    def __len__(self):
        return len(self.input_images)
    
    def read_img(self, filename):
        # Convert to RGB for scene flow finalpass data
        img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img_path = os.path.join(self.root_dir, 'Input', self.input_images[idx])
        gt_img_path = os.path.join(self.root_dir, 'GT', self.gt_images[idx])

        input_image = self.read_img(input_img_path)
        gt_image = self.read_img(gt_img_path)

        sample = {'input': input_image, 'gt': gt_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size

# Example usage
# dataset = CustomDataset(root_dir='path/to/tgt', transform=your_transforms)
# sample = dataset[0]
