import cv2
import numpy as np
from PIL import Image
import os
import torch
import torchvision

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

class PnemoImgMask(object):
    def __init__(self, root,  transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "SIIM_png_train"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask_new3"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "SIIM_png_train", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask_new3", self.masks[idx])
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        if np.amax(mask)==0:
            image_id = torch.tensor([0])
        else:
            image_id = torch.tensor([1])
        image_id = image_id.type(torch.LongTensor)
        sample={"image": img, "mask": mask}
        if self.transforms is not None:
            sample = self.transforms(sample)

        target={}
        target['mask'] = sample['mask']
        target['image id'] = image_id

        return {"image": sample['image'], "target": target}

    def __len__(self):
        return len(self.imgs)

class Preprocess(object):
    """Preprocessing data using opencv """
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["mask"]
        inv_img = 255-img
        ret,thresh3 = cv2.threshold(inv_img,70,255,cv2.THRESH_TRUNC)
        kernel = np.ones((20,20),np.uint8)
        closing = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel)
        final = cv2.subtract(inv_img, closing)
        return {"image":final,
                "mask":mask}

class Gray2RGB(object):
    """Convert 2darrays to 3darrays by adding channels."""
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        image_resize = cv2.resize(image,(224,224))
        mask_resize = cv2.resize(mask,(224,224))
        imageRGB = cv2.cvtColor(image_resize,cv2.COLOR_GRAY2RGB)
        maskRGB = cv2.cvtColor(mask_resize,cv2.COLOR_GRAY2RGB)
        return {"image":imageRGB,
                "mask":maskRGB}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors
        Permutes the dimensions from [0,1,2] to [2,0,1]"""
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        return {"image":torch.from_numpy(image).permute(2,0,1).type(torch.FloatTensor),
                "mask":torch.from_numpy(mask).permute(2,0,1).type(torch.FloatTensor)}

if __name__ == "__main__":
    root = "/media/arshita/Windows/Users/arshi/Desktop/Project2"
    transformed_dataset = PnemoImgMask(root,
                transforms=transforms.Compose([Preprocess(),
                                                Gray2RGB(),
                                                ToTensor()]))
    print(len(transformed_dataset))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]; print(i, sample['image'].shape)
        target = sample['target']; print(i, target['image id'])
        if target['image id']==1: break
