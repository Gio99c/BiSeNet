from unicodedata import name
import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice, map_label, one_hot
import random
import matplotlib.pyplot as plt 
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from pathlib import Path
import json
import torchvision.transforms.functional as F


class Cityscapes(Dataset):
    
    def __init__(self, root, image_folder, label_folder, images_names_file, json_file, image_size, loss='crossentropy', train = True):
        """
        Parameters:
            root: root folder including image_folder, label_folder, txt files and json file
            image_folder: name of the folder containing the images
            label_folder: name of the fodler containing the labels
            images_names_file: name of the txt file containing the list of the names of the images 
            jsonf_file: name of the json file containg infor about label mapping
            image_size: tuple, indicates (H,W) of the images
            loss: indicates the loss selected for the training, it usefull for the label encoding
            train: boolean, indicates if the dataset is prepared for the trainin phase or for the test phase
        Description:
            initialization of the parameters
        """
        super().__init__()
        self.root = root
        
        
        #Prepare the images
        images_folder_path = os.path.join(root, image_folder)
        images_path_list = [path for path in sorted(Path(images_folder_path).glob("*"))] #all the images contained in the image_folder

        images_names_file_path = os.path.join(root, images_names_file)
        images_names_list = [name.split("/")[1] for name in np.loadtxt(images_names_file_path, dtype="unicode")] #all the images name that have to be selected for the dataset

        self.images_list = [img for img in images_path_list if str(img).split("/")[-1] in images_names_list]
        
        #Prepare the labels
        labels_folder_path = os.path.join(root, label_folder)
        labels_path_list = [path for path in sorted(Path(labels_folder_path).glob("*"))]

        self.labels_list = [name for name in labels_path_list if str(name).split("/")[-1].replace("_gtFine_labelIds.png", "_leftImg8bit.png") in images_names_list]

        """
        print(len(self.images_list))
        print(len(self.labels_list))
        print(self.labels_list[0])

        label = Image.open(self.labels_list[0])
        label = np.array(label)
        #print(np.unique(label))
        """



        #Prepare the label mapping info
        f = open(os.path.join(root, json_file))
        info = json.load(f)
        self.label_mapping_info = {el[0]:el[1] for el in info['label2train']}
        self.mean = np.array(info['mean'])
        f.close()

        self.image_size = image_size
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]  # as indicated in the paper
        self.loss = loss
        self.train = train
        self.to_tensor = transforms.Compose([
            transforms.Normalize((73.158359210711552,82.908917542625858,72.392398761941593), (47.675755341814678, 48.494214368814916, 47.736546325441594)),                                               #each image is transformed into a tensor and normalized
            transforms.ToTensor(),
        ])

    def __len__(self):
        """
        Description:
            return the number of images inside the dataset
        """
        return len(self.images_list)
    
    def __getitem__(self, index):
        
        seed = random.random() #seed for random cropping

        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))
        #print("scale factor: ", scale)
        
        img = Image.open(self.images_list[index])


        
        # randomly resize image and random crop
        # =====================================
        if self.train:
            img = transforms.Resize(scale, F.InterpolationMode.BILINEAR)(img)         #rescale the image
            #print("Images's dimension after resize: ", img.size)
            #img.show()
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)          #take a piece of the image => this is the returned image
            #print(f"Image's dimension after the crop transformation: {img.size}")
            #img.show()       
        # =====================================
        

        label = Image.open(self.labels_list[index])
        label = Image.fromarray(map_label(np.array(label), self.label_mapping_info))

        
        # randomly resize label and random crop
        # =====================================
        if self.train:
            label = transforms.Resize(scale, F.InterpolationMode.NEAREST)(label)        #same steps of the image
            #print(f"Label's dimension after the resize: {label.size}")
            #label.show()
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)        #the seed guarantees that the two portions are the same 
            #print(f"Label's dimension after cropping: {label.size}")
            #label.show()
        # ===================================== 
        

        img = np.array(img, dtype = np.float32)
        img -= self.mean

        label = np.array(label, dtype = np.float32)
        
        """ decommentare se si vuole inserire la possibilità di flippare l'immagine
        if self.train:
            seq_det = self.fliplr.to_deterministic() 

            img = seq_det.augment_image(img)
            print(f"Image dimension after the flip: {img.shape}")
            label = seq_det.augment_image(label)
            print(f"Label dimension after the flip: {label.shape}")
        """

        # image: from [H, W, C] -> [C, H, W]
        """ print(img.shape)
        img -= self.mean
        print(img.shape)
        img = Image.fromarray(img)
        #img = transforms.ToTensor()(img)
        img = torch.tensor(img) """

        
        """  img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        print(np.unique(img))
        
        label = torch.from_numpy(label).long()
        print(np.unique(label)) """

        img = img.transpose([2,0,1])

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        

        return img, label



        


            




    





if __name__ =='__main__':
    root = 'data/Cityscapes'
    image_folder = 'images'
    label_folder = 'labels'
    images_names_file = 'train.txt'
    json_file = 'info.json'
    image_size = (512,1024)
    loss = 'dice'
    dataset = Cityscapes(root, image_folder, label_folder, images_names_file, json_file, image_size, loss)

    img, label = dataset[1]

    print(img.size())
    print(type(label))

    image = transforms.ToPILImage()(img)
    image.show()

    label = transforms.ToPILImage()(label)
    label.show()






