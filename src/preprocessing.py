import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets,transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

img_size=128
batch_size=64


transfrom=transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


def remove_corrupted_file(dir):
    for sub_dir in os.listdir(dir):
        for file in os.listdir(os.path.join(dir,sub_dir)):
            try:
                Image.open(os.path.join(dir,sub_dir,file))
            except:
                print("remove corrupted file:",os.path.join(dir,sub_dir,file))
                os.remove(os.path.join(dir,sub_dir,file))


def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def preprocess(root_dir="data"):
    train_dir=os.path.join(root_dir,"train")
    valid_dir=os.path.join(root_dir,"valid")

    train_dataset=datasets.ImageFolder(train_dir,transform=transfrom)
    valid_dataset=datasets.ImageFolder(valid_dir,transform=transfrom)

    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=True)

    return train_loader,valid_loader


def main():
    remove_corrupted_file("data/train")
    remove_corrupted_file("data/valid")

    train_loader,valid_loader=preprocess()
    print("train_loader:",len(train_loader))
    print("valid_loader:",len(valid_loader))


    #testing
    data_iter=iter(train_loader)
    images,labels=next(data_iter)
    imshow(torchvision.utils.make_grid(images[:8]))


if __name__=="__main__":
    main()