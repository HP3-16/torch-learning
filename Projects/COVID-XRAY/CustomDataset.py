import os
import glob

import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset

class CustomImgDataset(Dataset):
    def __init__(self,root,transforms):
        self.root = root
        self.transforms = transforms

        self.CLASS_FOLDERS = [f for f in os.listdir(root)]
        self.IMG_PATHS = []
        self.IMG_LABELS = []

        for label, folder in enumerate(self.CLASS_FOLDERS):
            curr_img_path =  glob.glob(os.path.join(root,folder,'*jpeg'))
            self.IMG_PATHS.extend(curr_img_path)
            self.IMG_LABELS.extend([label]*len(curr_img_path))
        
    def __getitem__(self, index):
        img_path = self.IMG_PATHS[index]
        label = self.IMG_LABELS[index]
        image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)

        return image, label
    
    def __len__(self):
        return len(self.IMG_PATHS)
