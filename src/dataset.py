from typing import Any
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os


class BSDDataset(Dataset):
    def __init__(self, 
                 root_path = 'data/HED-BSDS', 
                 train = True,
                 img_size = (256,256)) -> None:
        super().__init__()

        self.root_path = root_path
        self.train = train
        self.img_size = img_size

        filename = 'train_pair.lst'
        with open(os.path.join(self.root_path, filename)) as f:
            lines = f.readlines()

        lines = [line.rstrip('\n').split(" ") for line in lines]
        lines = lines[: int(0.8*len(lines))] if self.train else lines[int(0.8*len(lines)):]
        self.list_im, self. list_gt = zip(*lines)

        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.list_im)
    
    def get_image(self, path: str, is_color = True):
        if self.root_path not in path:
            path = self.root_path +"/"+ path

        flag = cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE

        img = cv2.imread(path, flag)
        img = cv2.resize(img, self.img_size)

        return img
    
    def __getitem__(self, index) -> Any:
        img = self.get_image(self.list_im[index])
        gt = self.get_image(self.list_gt[index], is_color= False)
        

        img = self.transform(img)
        gt = self.transform(gt)

        return img, gt








def get_bsd_loader(root_path = 'data/HED-BSDS', batch_size = 32, img_size = (256, 256)):
    trainset = BSDDataset(root_path=root_path, img_size= img_size)
    valset = BSDDataset(root_path=root_path, img_size= img_size, train= False)

    train_loader = DataLoader(trainset, batch_size= batch_size, shuffle= True, num_workers= 6, persistent_workers=True)
    val_loader = DataLoader(valset, batch_size= batch_size, shuffle= False, num_workers= 6, persistent_workers=True)

    return train_loader, val_loader