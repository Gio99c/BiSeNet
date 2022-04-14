import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision
import json

class Map:
    """
    Maps every pixel to the respective object in the dictionary
    Input:
        mapper: dict, dictionary of the mapping
    """
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, input):
        return np.vectorize(self.mapper.__getitem__)(input)

class ToTensor:
    """
    Convert into a tensor of float32: differently from transforms.ToTensor() this function does not normalize the values in [0,1] and does not swap the dimensions
    """
    def __call__(self, input):
        return torch.as_tensor(input, dtype=torch.float32)

# Don't know if it will be useful or if we will subtract the mean inside the dataset class
class MeanSubtraction:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, input):
        return input - self.mean


class CustomDataset(VisionDataset):
    def __init__(self, root, list_path, image_folder, labels_folder, max_iters=None, info_path=None,  transform=ToTensor(), target_transform=ToTensor()):
        """
        Inputs:
            root: string, path of the root folder where images and labels are stored
            list_path: string, path of the file used to split the dataset (train.txt/val.txt)
            image_folder: string, path of the images folder
            labels_folder: string, path of the labels folder
            transform: transformation to be applied on the images
            target_transform: transformation to be applied on the labels

        self.images = list containing the paths of the images 
        self.labels = list contating the paths of the labels
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.list_path = list_path                              # path to train.txt/val.txt
        self.max_iters = max_iters                              # maximum number of iteration
        self.info_path = info_path                              # path to the descriptor file info.json
        images_folder_path = Path(self.root) / image_folder     # absolute path of the folder containing the images
        labels_folder_path = Path(self.root) / labels_folder    # absolute path of the folder containing the labels
        

        #Retrive the file names of the images and labels contained in the indicated folders
        image_name_list = np.array(sorted(images_folder_path.glob("*")))
        labels_list = np.array(sorted(labels_folder_path.glob("*")))

        #Prepare lists of data and labels
        name_samples = [l.split("/")[1] for l in np.loadtxt(f"{root}/{list_path}", dtype="unicode")] # creates the list of the images names for the train/validation according to list_path
        self.images = [img for img in image_name_list if str(img).split("/")[-1] in name_samples]    # creates the list of images names filtered according to name_samples
        self.labels = [img for img in labels_list if str(img).split("/")[-1].replace("_gtFine_labelIds.png", "_leftImg8bit.png") in name_samples]  # creates the list of label image names filtered according to name_samples
        


    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]

        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))

        if self.transforms:
            image = self.transform(image)           # applies the transforms for the images
            label = self.target_transform(label)    # applies the transforms for the labels

        return image, label


if __name__ == "__main__":
    info = json.load(open("/Users/gio/Documents/GitHub/BiSeNet/data/Cityscapes/info.json"))
    mapper = dict(info["label2train"])
    crop_width = 1024
    crop_height = 512
    composed = torchvision.transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(0, scale=(0.75, 2.)), transforms.RandomCrop((crop_height, crop_width), pad_if_needed=True)])
    composed_target = torchvision.transforms.Compose([Map(mapper), ToTensor()])
    data = CustomDataset("./data/Cityscapes", "train.txt", "images/", "labels/", transform=composed,target_transform=composed_target)
    image, label = data[30]

    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(image.permute(1,2,0))
    axs[1].imshow(label.numpy().astype("uint8"))
    plt.show()
    

    ## It works, but we should find a way to apply the transformation on both the image AND the respective label.
    ## One possible solution could be https://stackoverflow.com/questions/65447992/pytorch-how-to-apply-the-same-random-transformation-to-multiple-image