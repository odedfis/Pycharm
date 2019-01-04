import cv2
import pandas as pd
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
import torch
data_dir = ''

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: randomRotate(x)),
        transforms.ToTensor(),
        transforms.Normalize([0.8240639, 0.64114773, 0.47744426], [0.03240008, 0.0981621, 0.12443555])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.8240639, 0.64114773, 0.47744426], [0.03240008, 0.0981621, 0.12443555])
    ])
}


def load_img(filename):
    img_path = filename
    with open(img_path, 'rb') as f:
        with Image.open(f) as img_f:
            return img_f.convert('RGB')

# define custom dataset
class MyDataSet(data.Dataset):
    def __init__(self, filename, training=True, validating=False, train_percent=0.85, transforms=None):
        df = pd.read_csv(filename)
        if training:
            split_index = (int)(df.values.shape[0]*train_percent)
            if validating:
                split_data = df.values[split_index:]
            else:
                split_data = df.values[:split_index]
            imgs = [None]*split_data.shape[0]
            labels = [None]*split_data.shape[0]
            for i, row in enumerate(split_data):
                fn, labels[i] = row
                imgs[i] = load_img(fn)
        else:
            imgs = [None]*df.values.shape[0]
            for i, row in enumerate(df.values):
                fn, _ = row
                imgs[i] = load_img(int(fn))
        self.imgs = imgs
        self.training = training
        self.transforms = transforms
        self.num = len(imgs)
        if self.training:
            self.labels = np.array(labels, dtype=np.float32)

    def __getitem__(self, index):
        if not self.transforms is None:
            img = self.transforms(self.imgs[index])
        if self.training:
            return img, self.labels[index]
        else:
            return img

    def __len__(self):
        return self.num


import random

def randomRotate(img):
    angel = random.randint(0, 4) * 90
    return img.rotate(angel)


def get_data_loader(filename='/home/odedf/learning/lw_data/csvfile.csv', training=True, validating=False, shuffle=True):
    if training and not validating:
        transkey = 'train'
    else:
        transkey = 'val'
    dset = MyDataSet(filename, training=training, validating=validating, transforms=data_transforms[transkey])
    loader = torch.utils.data.DataLoader(dset, batch_size=8, shuffle=shuffle, num_workers=10)
    loader.num = dset.num
    return loader

if __name__ == '__main__':
    dset = MyDataSet('/home/odedf/learning/lw_data/csvfile.csv', transforms=transforms.Compose([transforms.ToTensor()]))  # PyTorch Dataset object
    dataloader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=4)
    images, labels = iter(dataloader).next()
    # images.shape = ( 32, 3, 80, 80)
    numpy_images = images.numpy()

    per_image_mean = np.mean(numpy_images, axis=(2, 3))  # Shape (32,3)
    per_image_std = np.std(numpy_images, axis=(2, 3))  # Shape (32,3)

    pop_channel_mean = np.mean(per_image_mean, axis=0)  # Shape (3,)
    pop_channel_std = np.mean(per_image_std, axis=0)  # Shape (3,)
    print('oded')
